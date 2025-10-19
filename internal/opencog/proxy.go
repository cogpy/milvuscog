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

package opencog

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus/pkg/v2/log"
	"go.uber.org/zap"
)

// CognitiveProxy provides cognitive query processing capabilities
type CognitiveProxy struct {
	coordinator *CognitiveCoordinator
	logger      *zap.Logger
}

// NewCognitiveProxy creates a new cognitive proxy
func NewCognitiveProxy(coordinator *CognitiveCoordinator) *CognitiveProxy {
	return &CognitiveProxy{
		coordinator: coordinator,
		logger:      log.L(),
	}
}

// CognitiveQueryRequest represents a cognitive query request
type CognitiveQueryRequest struct {
	CollectionName   string                 `json:"collection_name"`
	QueryVector      []float32              `json:"query_vector"`
	TopK             int                    `json:"top_k"`
	ConceptFilters   []string               `json:"concept_filters"`
	ReasoningEnabled bool                   `json:"reasoning_enabled"`
	IntentQuery      string                 `json:"intent_query"`
	ContextFilters   map[string]interface{} `json:"context_filters"`
	AttentionFocus   []string               `json:"attention_focus"`
}

// CognitiveQueryResponse represents a cognitive query response
type CognitiveQueryResponse struct {
	QueryID          string                  `json:"query_id"`
	Results          []*CognitiveSearchHit   `json:"results"`
	ReasoningSteps   []*ReasoningStep        `json:"reasoning_steps"`
	InferredConcepts []string                `json:"inferred_concepts"`
	AttentionMap     map[string]int16        `json:"attention_map"`
	ExecutionTime    int64                   `json:"execution_time_ms"`
}

// ProcessCognitiveQuery processes a cognitive query with intent understanding
func (cp *CognitiveProxy) ProcessCognitiveQuery(ctx context.Context, req *CognitiveQueryRequest) (*CognitiveQueryResponse, error) {
	startTime := getCurrentTimestamp()
	
	cp.logger.Debug("Processing cognitive query",
		zap.String("collection", req.CollectionName),
		zap.Int("top_k", req.TopK),
		zap.String("intent", req.IntentQuery),
		zap.Bool("reasoning", req.ReasoningEnabled))
	
	// Find collection ID by name
	collectionID, err := cp.findCollectionIDByName(req.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("failed to find collection: %w", err)
	}
	
	// Apply attention focus if specified
	if len(req.AttentionFocus) > 0 {
		err := cp.coordinator.attentionAllocation.FocusAttention(req.AttentionFocus, 2)
		if err != nil {
			cp.logger.Error("Failed to apply attention focus", zap.Error(err))
		}
	}
	
	// Process intent query to extract additional concepts
	additionalConcepts, err := cp.processIntentQuery(ctx, req.IntentQuery)
	if err != nil {
		cp.logger.Error("Failed to process intent query", zap.Error(err))
		additionalConcepts = []string{}
	}
	
	// Combine concept filters
	allConceptFilters := append(req.ConceptFilters, additionalConcepts...)
	
	// Perform cognitive search
	searchResult, err := cp.coordinator.CognitiveSearch(
		ctx,
		collectionID,
		req.QueryVector,
		req.TopK,
		allConceptFilters,
		req.ReasoningEnabled,
	)
	if err != nil {
		return nil, fmt.Errorf("cognitive search failed: %w", err)
	}
	
	// Build attention map
	attentionMap := cp.buildAttentionMap(searchResult.Results)
	
	// Build response
	response := &CognitiveQueryResponse{
		QueryID:          searchResult.QueryID,
		Results:          searchResult.Results,
		ReasoningSteps:   searchResult.Reasoning,
		InferredConcepts: additionalConcepts,
		AttentionMap:     attentionMap,
		ExecutionTime:    getCurrentTimestamp() - startTime,
	}
	
	cp.logger.Debug("Cognitive query processed",
		zap.String("query_id", response.QueryID),
		zap.Int("results", len(response.Results)),
		zap.Int("reasoning_steps", len(response.ReasoningSteps)),
		zap.Int64("execution_time_ms", response.ExecutionTime))
	
	return response, nil
}

// processIntentQuery processes natural language intent to extract concepts
func (cp *CognitiveProxy) processIntentQuery(ctx context.Context, intentQuery string) ([]string, error) {
	if intentQuery == "" {
		return []string{}, nil
	}
	
	// Simple intent processing - in practice, this would use NLP
	concepts := make([]string, 0)
	
	// Extract keywords (simplified approach)
	keywords := extractKeywords(intentQuery)
	
	// Map keywords to concepts in the AtomSpace
	for _, keyword := range keywords {
		// Look for concept nodes with similar names
		conceptAtoms := cp.coordinator.atomSpace.GetAtomsByType(ConceptNode)
		for _, conceptAtom := range conceptAtoms {
			if matchesKeyword(conceptAtom.Name, keyword) {
				concepts = append(concepts, conceptAtom.Name)
			}
		}
	}
	
	cp.logger.Debug("Processed intent query",
		zap.String("intent", intentQuery),
		zap.Strings("extracted_concepts", concepts))
	
	return concepts, nil
}

// CognitiveInsertRequest represents a cognitive insert request
type CognitiveInsertRequest struct {
	CollectionName string                 `json:"collection_name"`
	Vectors        [][]float32            `json:"vectors"`
	Concepts       [][]string             `json:"concepts"`
	Relations      []map[string]string    `json:"relations"`
	TruthValues    []*TruthValue          `json:"truth_values"`
	Metadata       []map[string]interface{} `json:"metadata"`
}

// CognitiveInsertResponse represents a cognitive insert response
type CognitiveInsertResponse struct {
	InsertedIDs   []string `json:"inserted_ids"`
	SuccessCount  int      `json:"success_count"`
	ErrorCount    int      `json:"error_count"`
	ExecutionTime int64    `json:"execution_time_ms"`
}

// ProcessCognitiveInsert processes a cognitive insert with concept and relation extraction
func (cp *CognitiveProxy) ProcessCognitiveInsert(ctx context.Context, req *CognitiveInsertRequest) (*CognitiveInsertResponse, error) {
	startTime := getCurrentTimestamp()
	
	cp.logger.Debug("Processing cognitive insert",
		zap.String("collection", req.CollectionName),
		zap.Int("vector_count", len(req.Vectors)))
	
	// Find collection ID by name
	collectionID, err := cp.findCollectionIDByName(req.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("failed to find collection: %w", err)
	}
	
	response := &CognitiveInsertResponse{
		InsertedIDs:   make([]string, 0),
		SuccessCount:  0,
		ErrorCount:    0,
		ExecutionTime: 0,
	}
	
	// Process each vector
	for i, vector := range req.Vectors {
		var concepts []string
		var relations map[string]string
		var truthValue *TruthValue
		
		// Get concepts for this vector
		if i < len(req.Concepts) {
			concepts = req.Concepts[i]
		}
		
		// Get relations for this vector
		if i < len(req.Relations) {
			relations = req.Relations[i]
		}
		
		// Get truth value for this vector
		if i < len(req.TruthValues) {
			truthValue = req.TruthValues[i]
		}
		
		// Extract additional concepts from metadata
		if i < len(req.Metadata) {
			additionalConcepts := cp.extractConceptsFromMetadata(req.Metadata[i])
			concepts = append(concepts, additionalConcepts...)
		}
		
		// Insert cognitive vector
		vectorID, err := cp.coordinator.InsertCognitiveVector(
			ctx,
			collectionID,
			vector,
			concepts,
			relations,
			truthValue,
		)
		
		if err != nil {
			cp.logger.Error("Failed to insert cognitive vector",
				zap.Int("vector_index", i),
				zap.Error(err))
			response.ErrorCount++
		} else {
			response.InsertedIDs = append(response.InsertedIDs, vectorID)
			response.SuccessCount++
		}
	}
	
	response.ExecutionTime = getCurrentTimestamp() - startTime
	
	cp.logger.Debug("Cognitive insert processed",
		zap.String("collection", req.CollectionName),
		zap.Int("success_count", response.SuccessCount),
		zap.Int("error_count", response.ErrorCount),
		zap.Int64("execution_time_ms", response.ExecutionTime))
	
	return response, nil
}

// CognitiveReasoningRequest represents a reasoning request
type CognitiveReasoningRequest struct {
	CollectionName string   `json:"collection_name"`
	Mode          string   `json:"mode"` // "forward", "backward", "pattern_match"
	Goal          *Atom    `json:"goal,omitempty"`
	Pattern       *Atom    `json:"pattern,omitempty"`
	MaxSteps      int      `json:"max_steps"`
	MaxDepth      int      `json:"max_depth"`
}

// CognitiveReasoningResponse represents a reasoning response
type CognitiveReasoningResponse struct {
	Mode           string           `json:"mode"`
	NewAtoms       []*Atom          `json:"new_atoms"`
	MatchedAtoms   []*Atom          `json:"matched_atoms"`
	ReasoningSteps []*ReasoningStep `json:"reasoning_steps"`
	ExecutionTime  int64            `json:"execution_time_ms"`
}

// ProcessCognitiveReasoning processes a cognitive reasoning request
func (cp *CognitiveProxy) ProcessCognitiveReasoning(ctx context.Context, req *CognitiveReasoningRequest) (*CognitiveReasoningResponse, error) {
	startTime := getCurrentTimestamp()
	
	cp.logger.Debug("Processing cognitive reasoning",
		zap.String("collection", req.CollectionName),
		zap.String("mode", req.Mode),
		zap.Int("max_steps", req.MaxSteps))
	
	response := &CognitiveReasoningResponse{
		Mode:           req.Mode,
		NewAtoms:       make([]*Atom, 0),
		MatchedAtoms:   make([]*Atom, 0),
		ReasoningSteps: make([]*ReasoningStep, 0),
		ExecutionTime:  0,
	}
	
	plnReasoner := NewPLNReasoner(cp.coordinator.reasoningEngine)
	
	switch req.Mode {
	case "forward":
		newAtoms, err := plnReasoner.ForwardChain(ctx, req.MaxSteps)
		if err != nil {
			return nil, fmt.Errorf("forward chaining failed: %w", err)
		}
		response.NewAtoms = newAtoms
		
	case "backward":
		if req.Goal == nil {
			return nil, fmt.Errorf("goal is required for backward chaining")
		}
		proof, err := plnReasoner.BackwardChain(ctx, req.Goal, req.MaxDepth)
		if err != nil {
			return nil, fmt.Errorf("backward chaining failed: %w", err)
		}
		response.MatchedAtoms = proof
		
	case "pattern_match":
		if req.Pattern == nil {
			return nil, fmt.Errorf("pattern is required for pattern matching")
		}
		matches, err := cp.coordinator.patternMatcher.MatchPattern(req.Pattern, nil)
		if err != nil {
			return nil, fmt.Errorf("pattern matching failed: %w", err)
		}
		response.MatchedAtoms = matches
		
	default:
		return nil, fmt.Errorf("unsupported reasoning mode: %s", req.Mode)
	}
	
	response.ExecutionTime = getCurrentTimestamp() - startTime
	
	cp.logger.Debug("Cognitive reasoning processed",
		zap.String("mode", req.Mode),
		zap.Int("new_atoms", len(response.NewAtoms)),
		zap.Int("matched_atoms", len(response.MatchedAtoms)),
		zap.Int64("execution_time_ms", response.ExecutionTime))
	
	return response, nil
}

// Helper methods

// findCollectionIDByName finds a collection ID by name
func (cp *CognitiveProxy) findCollectionIDByName(name string) (string, error) {
	cp.coordinator.mutex.RLock()
	defer cp.coordinator.mutex.RUnlock()
	
	for id, collection := range cp.coordinator.collections {
		if collection.Name == name {
			return id, nil
		}
	}
	
	return "", fmt.Errorf("collection with name '%s' not found", name)
}

// buildAttentionMap builds a map of attention values from search results
func (cp *CognitiveProxy) buildAttentionMap(results []*CognitiveSearchHit) map[string]int16 {
	attentionMap := make(map[string]int16)
	
	for _, result := range results {
		if result.AttentionValue != nil {
			attentionMap[result.AtomID] = result.AttentionValue.STI
		}
	}
	
	return attentionMap
}

// extractConceptsFromMetadata extracts concepts from metadata
func (cp *CognitiveProxy) extractConceptsFromMetadata(metadata map[string]interface{}) []string {
	concepts := make([]string, 0)
	
	// Look for concept-related fields in metadata
	if tags, exists := metadata["tags"]; exists {
		if tagSlice, ok := tags.([]interface{}); ok {
			for _, tag := range tagSlice {
				if tagStr, ok := tag.(string); ok {
					concepts = append(concepts, tagStr)
				}
			}
		}
	}
	
	if category, exists := metadata["category"]; exists {
		if categoryStr, ok := category.(string); ok {
			concepts = append(concepts, categoryStr)
		}
	}
	
	if keywords, exists := metadata["keywords"]; exists {
		if keywordSlice, ok := keywords.([]interface{}); ok {
			for _, keyword := range keywordSlice {
				if keywordStr, ok := keyword.(string); ok {
					concepts = append(concepts, keywordStr)
				}
			}
		}
	}
	
	return concepts
}

// Utility functions

// extractKeywords extracts keywords from text (simplified)
func extractKeywords(text string) []string {
	// This is a very simplified implementation
	// In practice, you would use proper NLP libraries
	words := make([]string, 0)
	currentWord := ""
	
	for _, char := range text {
		if char == ' ' || char == ',' || char == '.' || char == '!' || char == '?' {
			if currentWord != "" {
				words = append(words, currentWord)
				currentWord = ""
			}
		} else {
			currentWord += string(char)
		}
	}
	
	if currentWord != "" {
		words = append(words, currentWord)
	}
	
	// Filter out common stop words
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "of": true, "with": true, "by": true, "from": true,
		"is": true, "are": true, "was": true, "were": true, "be": true,
		"been": true, "being": true, "have": true, "has": true, "had": true,
	}
	
	keywords := make([]string, 0)
	for _, word := range words {
		if !stopWords[word] && len(word) > 2 {
			keywords = append(keywords, word)
		}
	}
	
	return keywords
}

// matchesKeyword checks if a concept name matches a keyword
func matchesKeyword(conceptName, keyword string) bool {
	// Simple substring matching (case-insensitive)
	conceptLower := toLower(conceptName)
	keywordLower := toLower(keyword)
	
	return contains(conceptLower, keywordLower) || contains(keywordLower, conceptLower)
}

// toLower converts string to lowercase (simplified)
func toLower(s string) string {
	result := ""
	for _, char := range s {
		if char >= 'A' && char <= 'Z' {
			result += string(char + 32)
		} else {
			result += string(char)
		}
	}
	return result
}

// contains checks if string contains substring (simplified)
func contains(s, substr string) bool {
	if len(substr) > len(s) {
		return false
	}
	
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	
	return false
}

// getCurrentTimestamp returns current timestamp in milliseconds
func getCurrentTimestamp() int64 {
	return time.Now().UnixNano() / int64(time.Millisecond)
}