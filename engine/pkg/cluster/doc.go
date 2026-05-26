// Package cluster provides shared data structures and rpcx service definitions
// for the C/S (Client/Server) architecture.
//
// This package is imported by both S端 (server) and C端 (agent) implementations.
// All communication between C端 and S端 uses rpcx with the interfaces defined here.
package cluster