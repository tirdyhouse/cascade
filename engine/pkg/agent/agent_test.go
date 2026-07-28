package agent

import (
	"errors"
	"testing"
	"time"
)

func TestCommandPollErrorLoggingIsRateLimitedAndResets(t *testing.T) {
	agent := New(&Config{})
	first := errors.New("control plane unavailable")
	base := time.Date(2026, time.July, 28, 0, 0, 0, 0, time.UTC)

	if !agent.shouldLogCommandPollError(base, first) {
		t.Fatal("first command poll error should be logged")
	}
	if agent.shouldLogCommandPollError(base.Add(time.Second), first) {
		t.Fatal("repeated command poll error should be rate limited")
	}
	if !agent.shouldLogCommandPollError(base.Add(2*time.Second), errors.New("different control plane error")) {
		t.Fatal("a changed command poll error should be logged")
	}
	if !agent.shouldLogCommandPollError(base.Add(commandPollErrorLogInterval+3*time.Second), errors.New("different control plane error")) {
		t.Fatal("the same command poll error should be logged again after the interval")
	}

	agent.resetCommandPollError()
	if !agent.shouldLogCommandPollError(base.Add(commandPollErrorLogInterval+4*time.Second), first) {
		t.Fatal("a successful poll should reset error logging")
	}
}
