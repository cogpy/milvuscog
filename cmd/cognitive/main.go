// Copyright (C) 2024 Milvus Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/milvus-io/milvus/internal/opencog"
)

// CognitiveServer provides HTTP API for cognitive operations
type CognitiveServer struct {
	coordinator *opencog.CognitiveCoordinator
	proxy       *opencog.CognitiveProxy
	router      *gin.Engine
	port        int
}

// NewCognitiveServer creates a new cognitive server
func NewCognitiveServer(port int) *CognitiveServer {
	coordinator := opencog.NewCognitiveCoordinator()
	proxy := opencog.NewCognitiveProxy(coordinator)
	
	router := gin.Default()
	
	server := &CognitiveServer{
		coordinator: coordinator,
		proxy:       proxy,
		router:      router,
		port:        port,
	}
	
	server.setupRoutes()
	return server
}

// setupRoutes configures the HTTP routes
func (cs *CognitiveServer) setupRoutes() {
	// Health check
	cs.router.GET("/health", cs.healthCheck)
	
	// Collection management
	cs.router.POST("/cognitive/collections", cs.createCollection)
	cs.router.GET("/cognitive/collections", cs.listCollections)
	cs.router.GET("/cognitive/collections/:name", cs.getCollection)
	
	// Vector operations
	cs.router.POST("/cognitive/collections/:name/vectors", cs.insertVectors)
	cs.router.POST("/cognitive/collections/:name/search", cs.searchVectors)
	
	// Reasoning operations
	cs.router.POST("/cognitive/collections/:name/reasoning", cs.performReasoning)
	cs.router.GET("/cognitive/collections/:name/atoms", cs.getAtoms)
	
	// AtomSpace operations
	cs.router.GET("/cognitive/atomspace/stats", cs.getAtomSpaceStats)
	cs.router.POST("/cognitive/atomspace/pattern", cs.matchPattern)
	cs.router.POST("/cognitive/atomspace/attention", cs.focusAttention)
}

// Start starts the cognitive server
func (cs *CognitiveServer) Start() error {
	err := cs.coordinator.Start()
	if err != nil {
		return fmt.Errorf("failed to start coordinator: %w", err)
	}
	
	log.Printf("Starting cognitive server on port %d", cs.port)
	return cs.router.Run(fmt.Sprintf(":%d", cs.port))
}

// Stop stops the cognitive server
func (cs *CognitiveServer) Stop() error {
	return cs.coordinator.Stop()
}

// HTTP Handlers

func (cs *CognitiveServer) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
	})
}

func (cs *CognitiveServer) createCollection(c *gin.Context) {
	var req struct {
		Name   string                    `json:"name"`
		Schema *opencog.CognitiveSchema `json:"schema"`
		Config *opencog.ReasoningConfig `json:"config,omitempty"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	ctx := context.Background()
	collection, err := cs.coordinator.CreateCognitiveCollection(ctx, req.Name, req.Schema, req.Config)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusCreated, collection)
}

func (cs *CognitiveServer) listCollections(c *gin.Context) {
	// Simple implementation - in practice would query the coordinator
	c.JSON(http.StatusOK, gin.H{
		"collections": []string{},
		"count":       0,
	})
}

func (cs *CognitiveServer) getCollection(c *gin.Context) {
	name := c.Param("name")
	// Simple implementation - in practice would query the coordinator  
	c.JSON(http.StatusNotFound, gin.H{"error": "collection not found: " + name})
}

func (cs *CognitiveServer) insertVectors(c *gin.Context) {
	collectionName := c.Param("name")
	
	var req opencog.CognitiveInsertRequest
	req.CollectionName = collectionName
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	ctx := context.Background()
	response, err := cs.proxy.ProcessCognitiveInsert(ctx, &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

func (cs *CognitiveServer) searchVectors(c *gin.Context) {
	collectionName := c.Param("name")
	
	var req opencog.CognitiveQueryRequest
	req.CollectionName = collectionName
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	ctx := context.Background()
	response, err := cs.proxy.ProcessCognitiveQuery(ctx, &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

func (cs *CognitiveServer) performReasoning(c *gin.Context) {
	collectionName := c.Param("name")
	
	var req opencog.CognitiveReasoningRequest
	req.CollectionName = collectionName
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	ctx := context.Background()
	response, err := cs.proxy.ProcessCognitiveReasoning(ctx, &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

func (cs *CognitiveServer) getAtoms(c *gin.Context) {
	collectionName := c.Param("name")
	atomTypeStr := c.Query("type")
	limitStr := c.Query("limit")
	
	limit := 100 // default
	if limitStr != "" {
		if parsedLimit, err := strconv.Atoi(limitStr); err == nil {
			limit = parsedLimit
		}
	}
	
	var atoms []*opencog.Atom
	
	if atomTypeStr != "" {
		atomType := parseAtomType(atomTypeStr)
		atoms = cs.coordinator.GetAtomSpace().GetAtomsByType(atomType)
	}
	
	if len(atoms) > limit {
		atoms = atoms[:limit]
	}
	
	c.JSON(http.StatusOK, gin.H{
		"collection": collectionName,
		"atoms":      atoms,
		"count":      len(atoms),
		"total":      cs.coordinator.GetAtomSpace().Size(),
	})
}

func (cs *CognitiveServer) getAtomSpaceStats(c *gin.Context) {
	atomSpace := cs.coordinator.GetAtomSpace()
	
	stats := gin.H{
		"total_atoms": atomSpace.Size(),
		"atoms_by_type": gin.H{
			"concept_nodes":    len(atomSpace.GetAtomsByType(opencog.ConceptNode)),
			"vector_nodes":     len(atomSpace.GetAtomsByType(opencog.VectorNode)),
			"similarity_links": len(atomSpace.GetAtomsByType(opencog.SimilarityLink)),
			"inheritance_links": len(atomSpace.GetAtomsByType(opencog.InheritanceLink)),
			"evaluation_links": len(atomSpace.GetAtomsByType(opencog.EvaluationLink)),
		},
		"timestamp": time.Now().Unix(),
	}
	
	c.JSON(http.StatusOK, stats)
}

func (cs *CognitiveServer) matchPattern(c *gin.Context) {
	var req struct {
		Pattern     *opencog.Atom                   `json:"pattern"`
		Constraints map[string]interface{}          `json:"constraints,omitempty"`
		Limit       int                            `json:"limit,omitempty"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if req.Limit == 0 {
		req.Limit = 50
	}
	
	// Convert constraints to proper format
	constraintFuncs := make(map[string]func(*opencog.Atom) bool)
	for variable := range req.Constraints {
		// Create a local copy to avoid closure issues
		v := variable
		constraintFuncs[v] = func(atom *opencog.Atom) bool {
			// Simple constraint evaluation - in practice this would be more sophisticated
			return true
		}
	}
	
	matches, err := cs.coordinator.GetPatternMatcher().MatchPattern(req.Pattern, constraintFuncs)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	if len(matches) > req.Limit {
		matches = matches[:req.Limit]
	}
	
	c.JSON(http.StatusOK, gin.H{
		"matches": matches,
		"count":   len(matches),
	})
}

func (cs *CognitiveServer) focusAttention(c *gin.Context) {
	var req struct {
		AtomIDs []string `json:"atom_ids"`
		Radius  int      `json:"radius"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if req.Radius == 0 {
		req.Radius = 2
	}
	
	err := cs.coordinator.GetAttentionAllocation().FocusAttention(req.AtomIDs, req.Radius)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"status":       "success",
		"focused_atoms": req.AtomIDs,
		"radius":       req.Radius,
	})
}

// Helper functions

func parseAtomType(typeStr string) opencog.AtomType {
	switch strings.ToLower(typeStr) {
	case "concept", "conceptnode":
		return opencog.ConceptNode
	case "vector", "vectornode":
		return opencog.VectorNode
	case "similarity", "similaritylink":
		return opencog.SimilarityLink
	case "inheritance", "inheritancelink":
		return opencog.InheritanceLink
	case "evaluation", "evaluationlink":
		return opencog.EvaluationLink
	case "predicate", "predicatenode":
		return opencog.PredicateNode
	case "list", "listlink":
		return opencog.ListLink
	default:
		return opencog.ConceptNode
	}
}

func main() {
	port := flag.Int("port", 8080, "Port to run the cognitive server on")
	flag.Parse()
	
	server := NewCognitiveServer(*port)
	
	if err := server.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}