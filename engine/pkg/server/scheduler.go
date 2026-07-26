package server

import (
	"errors"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"predict/engine/pkg/cluster"
)

const defaultVLLMPort = 8000

const gatewayFailureCooldown = 5 * time.Second

var (
	// ErrNoGatewayCandidate means no healthy vLLM process currently serves the
	// requested model.
	ErrNoGatewayCandidate = errors.New("no eligible running vLLM node for requested model")
	// ErrGatewayCapacity means healthy model replicas exist but the gateway has
	// reached its per-node admission limit.
	ErrGatewayCapacity = errors.New("all eligible vLLM nodes are at gateway capacity")
)

// gatewayCandidate is an immutable registry snapshot used for a single
// scheduling decision.
type gatewayCandidate struct {
	NodeID       string
	IP           string
	VLLMPort     int
	ModelName    string
	GPUUtil      float64
	QueueLen     int32
	CacheHitRate float64
	CacheMode    cluster.CacheMode
}

// GatewayTarget identifies the selected vLLM endpoint. It is intentionally a
// value snapshot so a heartbeat cannot change an in-flight request's target.
type GatewayTarget struct {
	NodeID   string
	IP       string
	VLLMPort int
	Model    string
}

// GatewayLease represents one gateway request admitted to a node. Release
// must be called after the proxied response body finishes or is canceled.
type GatewayLease struct {
	Target GatewayTarget

	scheduler *GatewayScheduler
	once      sync.Once
}

// Release returns the admission slot. It is safe to call more than once.
func (l *GatewayLease) Release() {
	if l == nil || l.scheduler == nil {
		return
	}
	l.once.Do(func() {
		l.scheduler.release(l.Target.NodeID)
	})
}

// MarkUnavailable excludes this endpoint for a short period after a transport
// failure. The failed request is not retried because a POST can be ambiguous.
func (l *GatewayLease) MarkUnavailable() {
	if l == nil || l.scheduler == nil {
		return
	}
	l.scheduler.markUnavailable(l.Target.NodeID)
}

// GatewayScheduler selects a running vLLM replica. Agent QueueLen represents
// work already known to vLLM; inFlight additionally covers traffic accepted by
// this gateway but not yet visible in the next heartbeat.
type GatewayScheduler struct {
	registry    *Registry
	maxInFlight int

	mu               sync.Mutex
	inFlight         map[string]int
	unavailableUntil map[string]time.Time
	next             uint64
}

func NewGatewayScheduler(registry *Registry, maxInFlightPerNode int) *GatewayScheduler {
	if maxInFlightPerNode <= 0 {
		maxInFlightPerNode = 16
	}
	return &GatewayScheduler{
		registry:         registry,
		maxInFlight:      maxInFlightPerNode,
		inFlight:         make(map[string]int),
		unavailableUntil: make(map[string]time.Time),
	}
}

// Acquire reserves a request slot on the best currently eligible replica.
func (s *GatewayScheduler) Acquire(model string) (*GatewayLease, error) {
	if s == nil || s.registry == nil {
		return nil, ErrNoGatewayCandidate
	}

	candidates := s.registry.gatewayCandidates(model, time.Now())
	if len(candidates) == 0 {
		return nil, ErrNoGatewayCandidate
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].NodeID < candidates[j].NodeID
	})

	s.mu.Lock()
	defer s.mu.Unlock()

	available := make([]gatewayCandidate, 0, len(candidates))
	hasUnquarantinedNode := false
	now := time.Now()
	for _, candidate := range candidates {
		if until, unavailable := s.unavailableUntil[candidate.NodeID]; unavailable {
			if now.Before(until) {
				continue
			}
			delete(s.unavailableUntil, candidate.NodeID)
		}
		hasUnquarantinedNode = true
		if s.inFlight[candidate.NodeID] < s.maxInFlight {
			available = append(available, candidate)
		}
	}
	if len(available) == 0 {
		if !hasUnquarantinedNode {
			return nil, ErrNoGatewayCandidate
		}
		return nil, ErrGatewayCapacity
	}

	// Rotation provides fair tie-breaking, while queue length and active GPU
	// work remain the dominant scheduling signals.
	start := int(s.next % uint64(len(available)))
	best := -1
	bestScore := math.Inf(1)
	for offset := 0; offset < len(available); offset++ {
		index := (start + offset) % len(available)
		candidate := available[index]
		score := s.score(candidate)
		if score < bestScore-0.000001 {
			best = index
			bestScore = score
		}
	}

	selected := available[best]
	s.next++
	s.inFlight[selected.NodeID]++
	return &GatewayLease{
		Target: GatewayTarget{
			NodeID:   selected.NodeID,
			IP:       selected.IP,
			VLLMPort: selected.VLLMPort,
			Model:    selected.ModelName,
		},
		scheduler: s,
	}, nil
}

func (s *GatewayScheduler) release(nodeID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inFlight[nodeID] <= 1 {
		delete(s.inFlight, nodeID)
		return
	}
	s.inFlight[nodeID]--
}

func (s *GatewayScheduler) markUnavailable(nodeID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.unavailableUntil[nodeID] = time.Now().Add(gatewayFailureCooldown)
}

func (s *GatewayScheduler) snapshot() (int, map[string]int, map[string]int64) {
	if s == nil {
		return 0, map[string]int{}, map[string]int64{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	inFlight := make(map[string]int, len(s.inFlight))
	for nodeID, count := range s.inFlight {
		inFlight[nodeID] = count
	}
	quarantined := make(map[string]int64, len(s.unavailableUntil))
	now := time.Now()
	for nodeID, until := range s.unavailableUntil {
		if now.Before(until) {
			quarantined[nodeID] = until.UnixNano()
		} else {
			delete(s.unavailableUntil, nodeID)
		}
	}
	return s.maxInFlight, inFlight, quarantined
}

func (s *GatewayScheduler) score(candidate gatewayCandidate) float64 {
	queue := float64(candidate.QueueLen)
	if queue < 0 {
		queue = 0
	}
	queue += float64(s.inFlight[candidate.NodeID])
	gpuUtil := clamp(candidate.GPUUtil, 0, 1)
	cacheHitRate := clamp(candidate.CacheHitRate, 0, 100)

	// A queued request is far more important than a small utilization or cache
	// history difference. The cache term is a tie-breaker only; it is not a
	// prompt-specific cache-match signal.
	return queue*100 + gpuUtil*10 - cacheHitRate/100
}

func modelsEquivalent(requested, loaded string) bool {
	requested = strings.TrimSpace(requested)
	loaded = strings.TrimSpace(loaded)
	if requested == "" {
		return loaded != ""
	}
	if loaded == "" {
		return false
	}
	if requested == loaded {
		return true
	}
	return filepath.Base(strings.TrimSuffix(requested, "/")) ==
		filepath.Base(strings.TrimSuffix(loaded, "/"))
}

func clamp(value, min, max float64) float64 {
	if math.IsNaN(value) {
		return min
	}
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
