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
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/milvus-io/milvus/pkg/v2/log"
	"go.uber.org/zap"
)

// CognitiveCoordinator manages the cognitive layer on top of Milvus
type CognitiveCoordinator struct {
	atomSpace          *AtomSpace
	reasoningEngine    *ReasoningEngine
	patternMatcher     *PatternMatcher
	attentionAllocation *AttentionAllocation
	collections        map[string]*CognitiveCollection
	mutex              sync.RWMutex
	logger             *zap.Logger
	ctx                context.Context
	cancel             context.CancelFunc
	running            bool
}

// CognitiveCollection represents a cognitive collection that extends Milvus collections
type CognitiveCollection struct {
	ID              string
	Name            string
	Schema          *CognitiveSchema
	AtomSpaceID     string
	ReasoningConfig *ReasoningConfig
	CreatedTime     time.Time
	UpdatedTime     time.Time
}

// CognitiveSchema defines the cognitive schema for a collection
type CognitiveSchema struct {
	VectorField     *CognitiveField
	ConceptFields   []*CognitiveField
	RelationFields  []*CognitiveField
	TruthValueField *CognitiveField
}

// CognitiveField represents a field with cognitive properties
type CognitiveField struct {
	Name         string
	Type         CognitiveFieldType
	Dimension    int
	AtomType     AtomType
	Constraints  map[string]interface{}
}

// CognitiveFieldType represents the type of cognitive field
type CognitiveFieldType int

const (
	VectorFieldType CognitiveFieldType = iota
	ConceptFieldType
	RelationFieldType
	TruthValueFieldType
	AttentionFieldType
)

// ReasoningConfig configures reasoning behavior for a collection
type ReasoningConfig struct {
	EnableForwardChaining  bool
	EnableBackwardChaining bool
	MaxInferenceDepth     int
	InferenceInterval     time.Duration
	AttentionThreshold    int16
	TruthValueThreshold   float32
}

// NewCognitiveCoordinator creates a new cognitive coordinator
func NewCognitiveCoordinator() *CognitiveCoordinator {
	ctx, cancel := context.WithCancel(context.Background())
	
	atomSpace := NewAtomSpace()
	reasoningEngine := NewReasoningEngine(atomSpace)
	patternMatcher := NewPatternMatcher(atomSpace)
	attentionAllocation := NewAttentionAllocation(atomSpace)
	
	return &CognitiveCoordinator{
		atomSpace:          atomSpace,
		reasoningEngine:    reasoningEngine,
		patternMatcher:     patternMatcher,
		attentionAllocation: attentionAllocation,
		collections:        make(map[string]*CognitiveCollection),
		logger:             log.L(),
		ctx:                ctx,
		cancel:             cancel,
		running:            false,
	}
}

// Start starts the cognitive coordinator
func (cc *CognitiveCoordinator) Start() error {
	cc.mutex.Lock()
	defer cc.mutex.Unlock()
	
	if cc.running {
		return fmt.Errorf("cognitive coordinator already running")
	}
	
	cc.running = true
	
	// Start background reasoning processes
	go cc.runPeriodicReasoning()
	go cc.runAttentionMaintenance()
	
	cc.logger.Info("Cognitive coordinator started")
	return nil
}

// Stop stops the cognitive coordinator
func (cc *CognitiveCoordinator) Stop() error {
	cc.mutex.Lock()
	defer cc.mutex.Unlock()
	
	if !cc.running {
		return fmt.Errorf("cognitive coordinator not running")
	}
	
	cc.cancel()
	cc.running = false
	
	cc.logger.Info("Cognitive coordinator stopped")
	return nil
}

// CreateCognitiveCollection creates a new cognitive collection
func (cc *CognitiveCoordinator) CreateCognitiveCollection(
	ctx context.Context, 
	name string, 
	schema *CognitiveSchema,
	config *ReasoningConfig,
) (*CognitiveCollection, error) {
	cc.mutex.Lock()
	defer cc.mutex.Unlock()
	
	collectionID := fmt.Sprintf("cog_col_%s_%d", name, time.Now().UnixNano())
	
	if config == nil {
		config = &ReasoningConfig{
			EnableForwardChaining:  true,
			EnableBackwardChaining: false,
			MaxInferenceDepth:     3,
			InferenceInterval:     time.Minute * 5,
			AttentionThreshold:    50,
			TruthValueThreshold:   0.7,
		}
	}
	
	collection := &CognitiveCollection{
		ID:              collectionID,
		Name:            name,
		Schema:          schema,
		AtomSpaceID:     fmt.Sprintf("atomspace_%s", collectionID),
		ReasoningConfig: config,
		CreatedTime:     time.Now(),
		UpdatedTime:     time.Now(),
	}
	
	cc.collections[collectionID] = collection
	
	// Create collection-specific atom space concepts
	err := cc.initializeCollectionConcepts(collection)
	if err != nil {
		delete(cc.collections, collectionID)
		return nil, fmt.Errorf("failed to initialize collection concepts: %w", err)
	}
	
	cc.logger.Info("Created cognitive collection",
		zap.String("collection_id", collectionID),
		zap.String("name", name))
	
	return collection, nil
}

// initializeCollectionConcepts creates initial concepts for a collection
func (cc *CognitiveCoordinator) initializeCollectionConcepts(collection *CognitiveCollection) error {
	// Create collection concept
	collectionConcept := &Atom{
		ID:   fmt.Sprintf("concept_%s", collection.ID),
		Type: ConceptNode,
		Name: fmt.Sprintf("Collection_%s", collection.Name),
		TruthValue: &TruthValue{
			Strength:   1.0,
			Confidence: 1.0,
		},
		AttentionValue: &AttentionValue{
			STI:  200,
			LTI:  10,
			VLTI: true,
		},
		Vector:      nil,
		OutgoingSet: make([]*Atom, 0),
		IncomingSet: make([]*Atom, 0),
	}
	
	err := cc.atomSpace.AddAtom(collectionConcept)
	if err != nil {
		return fmt.Errorf("failed to add collection concept: %w", err)
	}
	
	// Create schema concepts
	for _, field := range collection.Schema.ConceptFields {
		fieldConcept := &Atom{
			ID:   fmt.Sprintf("concept_%s_%s", collection.ID, field.Name),
			Type: ConceptNode,
			Name: fmt.Sprintf("Field_%s", field.Name),
			TruthValue: &TruthValue{
				Strength:   1.0,
				Confidence: 1.0,
			},
			AttentionValue: &AttentionValue{
				STI:  150,
				LTI:  5,
				VLTI: false,
			},
			Vector:      nil,
			OutgoingSet: make([]*Atom, 0),
			IncomingSet: make([]*Atom, 0),
		}
		
		err := cc.atomSpace.AddAtom(fieldConcept)
		if err != nil {
			return fmt.Errorf("failed to add field concept: %w", err)
		}
		
		// Create inheritance link from field to collection
		inheritanceLink := cc.atomSpace.createInheritanceLink(
			fieldConcept,
			collectionConcept,
			1.0,
		)
		
		err = cc.atomSpace.AddAtom(inheritanceLink)
		if err != nil {
			return fmt.Errorf("failed to add inheritance link: %w", err)
		}
	}
	
	return nil
}

// InsertCognitiveVector inserts a vector with cognitive properties
func (cc *CognitiveCoordinator) InsertCognitiveVector(
	ctx context.Context,
	collectionID string,
	vector []float32,
	concepts []string,
	relations map[string]string,
	truthValue *TruthValue,
) (string, error) {
	cc.mutex.RLock()
	_, exists := cc.collections[collectionID]
	cc.mutex.RUnlock()
	
	if !exists {
		return "", fmt.Errorf("cognitive collection %s not found", collectionID)
	}
	
	vectorID := fmt.Sprintf("vec_%s_%d", collectionID, time.Now().UnixNano())
	
	// Create vector atom
	vectorAtom := cc.atomSpace.CreateVectorAtom(vectorID, fmt.Sprintf("Vector_%s", vectorID), vector, truthValue)
	
	err := cc.atomSpace.AddAtom(vectorAtom)
	if err != nil {
		return "", fmt.Errorf("failed to add vector atom: %w", err)
	}
	
	// Create concept atoms and links
	for _, conceptName := range concepts {
		conceptID := fmt.Sprintf("concept_%s_%s", collectionID, conceptName)
		conceptAtom := &Atom{
			ID:   conceptID,
			Type: ConceptNode,
			Name: conceptName,
			TruthValue: &TruthValue{
				Strength:   0.9,
				Confidence: 0.8,
			},
			AttentionValue: &AttentionValue{
				STI:  100,
				LTI:  0,
				VLTI: false,
			},
			Vector:      nil,
			OutgoingSet: make([]*Atom, 0),
			IncomingSet: make([]*Atom, 0),
		}
		
		// Try to get existing concept or add new one
		if existingConcept, exists := cc.atomSpace.GetAtom(conceptID); exists {
			conceptAtom = existingConcept
		} else {
			err := cc.atomSpace.AddAtom(conceptAtom)
			if err != nil {
				cc.logger.Error("Failed to add concept atom", zap.Error(err))
				continue
			}
		}
		
		// Create evaluation link between vector and concept
		evaluationLink := &Atom{
			ID:   fmt.Sprintf("eval_%s_%s", vectorID, conceptID),
			Type: EvaluationLink,
			Name: fmt.Sprintf("evaluation_%s_%s", vectorAtom.Name, conceptAtom.Name),
			TruthValue: &TruthValue{
				Strength:   0.8,
				Confidence: 0.7,
			},
			AttentionValue: &AttentionValue{
				STI:  75,
				LTI:  0,
				VLTI: false,
			},
			OutgoingSet: []*Atom{vectorAtom, conceptAtom},
			IncomingSet: make([]*Atom, 0),
		}
		
		err = cc.atomSpace.AddAtom(evaluationLink)
		if err != nil {
			cc.logger.Error("Failed to add evaluation link", zap.Error(err))
		}
	}
	
	// Create relation links
	for relationType, targetID := range relations {
		targetAtom, exists := cc.atomSpace.GetAtom(targetID)
		if !exists {
			cc.logger.Warn("Target atom for relation not found", 
				zap.String("target_id", targetID),
				zap.String("relation_type", relationType))
			continue
		}
		
		relationLink := &Atom{
			ID:   fmt.Sprintf("rel_%s_%s_%s", relationType, vectorID, targetID),
			Type: SimilarityLink, // Using similarity link for general relations
			Name: fmt.Sprintf("%s_%s_%s", relationType, vectorAtom.Name, targetAtom.Name),
			TruthValue: &TruthValue{
				Strength:   0.7,
				Confidence: 0.6,
			},
			AttentionValue: &AttentionValue{
				STI:  60,
				LTI:  0,
				VLTI: false,
			},
			OutgoingSet: []*Atom{vectorAtom, targetAtom},
			IncomingSet: make([]*Atom, 0),
		}
		
		err = cc.atomSpace.AddAtom(relationLink)
		if err != nil {
			cc.logger.Error("Failed to add relation link", zap.Error(err))
		}
	}
	
	// Update attention for the new vector
	err = cc.attentionAllocation.UpdateAttention(vectorID, 100)
	if err != nil {
		cc.logger.Error("Failed to update attention for new vector", zap.Error(err))
	}
	
	cc.logger.Debug("Inserted cognitive vector",
		zap.String("collection_id", collectionID),
		zap.String("vector_id", vectorID),
		zap.Int("vector_dim", len(vector)),
		zap.Int("concepts", len(concepts)),
		zap.Int("relations", len(relations)))
	
	return vectorID, nil
}

// CognitiveSearch performs cognitive search with reasoning
func (cc *CognitiveCoordinator) CognitiveSearch(
	ctx context.Context,
	collectionID string,
	queryVector []float32,
	topK int,
	conceptFilters []string,
	reasoningEnabled bool,
) (*CognitiveSearchResult, error) {
	cc.mutex.RLock()
	collection, exists := cc.collections[collectionID]
	cc.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("cognitive collection %s not found", collectionID)
	}
	
	result := &CognitiveSearchResult{
		QueryID:     fmt.Sprintf("query_%d", time.Now().UnixNano()),
		CollectionID: collectionID,
		TopK:        topK,
		Results:     make([]*CognitiveSearchHit, 0),
		Reasoning:   make([]*ReasoningStep, 0),
	}
	
	// Get all vector atoms in the collection
	vectorAtoms := cc.atomSpace.GetAtomsByType(VectorNode)
	
	// Filter by collection and concepts
	candidateAtoms := make([]*Atom, 0)
	for _, atom := range vectorAtoms {
		if cc.belongsToCollection(atom, collectionID) {
			if len(conceptFilters) == 0 || cc.matchesConceptFilters(atom, conceptFilters) {
				candidateAtoms = append(candidateAtoms, atom)
			}
		}
	}
	
	// Calculate similarities and rank results
	hits := make([]*CognitiveSearchHit, 0)
	for _, atom := range candidateAtoms {
		similarity := cc.calculateVectorSimilarity(queryVector, atom.Vector)
		
		hit := &CognitiveSearchHit{
			AtomID:     atom.ID,
			VectorID:   atom.ID,
			Similarity: similarity,
			Vector:     atom.Vector,
			TruthValue: atom.TruthValue,
			AttentionValue: atom.AttentionValue,
			Concepts:   cc.getAtomConcepts(atom),
			Relations:  cc.getAtomRelations(atom),
		}
		
		hits = append(hits, hit)
	}
	
	// Sort by similarity using Go's built-in sort
	sort.Slice(hits, func(i, j int) bool {
		return hits[i].Similarity > hits[j].Similarity
	})
	
	// Take top K
	if len(hits) > topK {
		hits = hits[:topK]
	}
	
	result.Results = hits
	
	// Perform reasoning if enabled
	if reasoningEnabled && collection.ReasoningConfig.EnableForwardChaining {
		reasoningSteps, err := cc.performSearchReasoning(ctx, result)
		if err != nil {
			cc.logger.Error("Failed to perform search reasoning", zap.Error(err))
		} else {
			result.Reasoning = reasoningSteps
		}
	}
	
	cc.logger.Debug("Cognitive search completed",
		zap.String("collection_id", collectionID),
		zap.String("query_id", result.QueryID),
		zap.Int("candidates", len(candidateAtoms)),
		zap.Int("results", len(result.Results)))
	
	return result, nil
}

// CognitiveSearchResult represents the result of a cognitive search
type CognitiveSearchResult struct {
	QueryID      string
	CollectionID string
	TopK         int
	Results      []*CognitiveSearchHit
	Reasoning    []*ReasoningStep
}

// CognitiveSearchHit represents a single search result
type CognitiveSearchHit struct {
	AtomID         string
	VectorID       string
	Similarity     float32
	Vector         []float32
	TruthValue     *TruthValue
	AttentionValue *AttentionValue
	Concepts       []string
	Relations      map[string][]string
}

// ReasoningStep represents a step in the reasoning process
type ReasoningStep struct {
	Rule        string
	Premises    []string
	Conclusion  string
	Confidence  float32
}

// calculateVectorSimilarity calculates cosine similarity between vectors
func (cc *CognitiveCoordinator) calculateVectorSimilarity(vec1, vec2 []float32) float32 {
	if len(vec1) != len(vec2) {
		return 0.0
	}
	
	var dotProduct, norm1, norm2 float32
	for i := 0; i < len(vec1); i++ {
		dotProduct += vec1[i] * vec2[i]
		norm1 += vec1[i] * vec1[i]
		norm2 += vec2[i] * vec2[i]
	}
	
	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}
	
	return dotProduct / (float32(math.Sqrt(float64(norm1))) * float32(math.Sqrt(float64(norm2))))
}

// Helper methods

// belongsToCollection checks if an atom belongs to a specific collection
func (cc *CognitiveCoordinator) belongsToCollection(atom *Atom, collectionID string) bool {
	// Check if atom ID contains collection ID prefix
	prefix := fmt.Sprintf("vec_%s_", collectionID)
	return strings.HasPrefix(atom.ID, prefix)
}

// matchesConceptFilters checks if an atom matches concept filters
func (cc *CognitiveCoordinator) matchesConceptFilters(atom *Atom, conceptFilters []string) bool {
	atomConcepts := cc.getAtomConcepts(atom)
	
	for _, filter := range conceptFilters {
		found := false
		for _, concept := range atomConcepts {
			if concept == filter {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	
	return true
}

// getAtomConcepts gets concepts associated with an atom
func (cc *CognitiveCoordinator) getAtomConcepts(atom *Atom) []string {
	concepts := make([]string, 0)
	
	for _, incoming := range atom.IncomingSet {
		if incoming.Type == EvaluationLink && len(incoming.OutgoingSet) >= 2 {
			if incoming.OutgoingSet[1].Type == ConceptNode {
				concepts = append(concepts, incoming.OutgoingSet[1].Name)
			}
		}
	}
	
	return concepts
}

// getAtomRelations gets relations associated with an atom
func (cc *CognitiveCoordinator) getAtomRelations(atom *Atom) map[string][]string {
	relations := make(map[string][]string)
	
	for _, incoming := range atom.IncomingSet {
		if incoming.Type == SimilarityLink && len(incoming.OutgoingSet) >= 2 {
			relationType := "similarity"
			targetID := ""
			
			if incoming.OutgoingSet[0].ID == atom.ID {
				targetID = incoming.OutgoingSet[1].ID
			} else {
				targetID = incoming.OutgoingSet[0].ID
			}
			
			if relations[relationType] == nil {
				relations[relationType] = make([]string, 0)
			}
			relations[relationType] = append(relations[relationType], targetID)
		}
	}
	
	return relations
}

// performSearchReasoning performs reasoning on search results
func (cc *CognitiveCoordinator) performSearchReasoning(ctx context.Context, result *CognitiveSearchResult) ([]*ReasoningStep, error) {
	steps := make([]*ReasoningStep, 0)
	
	// Simple example: infer similarity between top results
	if len(result.Results) >= 2 {
		atom1, exists1 := cc.atomSpace.GetAtom(result.Results[0].AtomID)
		atom2, exists2 := cc.atomSpace.GetAtom(result.Results[1].AtomID)
		
		if exists1 && exists2 {
			similarity := cc.calculateVectorSimilarity(atom1.Vector, atom2.Vector)
			
			if similarity > 0.8 {
				step := &ReasoningStep{
					Rule:       "High similarity inference",
					Premises:   []string{atom1.ID, atom2.ID},
					Conclusion: fmt.Sprintf("Atoms %s and %s are highly similar", atom1.ID, atom2.ID),
					Confidence: similarity,
				}
				steps = append(steps, step)
				
				// Create similarity link if it doesn't exist
				linkID := fmt.Sprintf("inferred_sim_%s_%s", atom1.ID, atom2.ID)
				if _, exists := cc.atomSpace.GetAtom(linkID); !exists {
					simLink := cc.atomSpace.CreateSimilarityLink(linkID, atom1, atom2, similarity)
					err := cc.atomSpace.AddAtom(simLink)
					if err != nil {
						cc.logger.Error("Failed to add inferred similarity link", zap.Error(err))
					}
				}
			}
		}
	}
	
	return steps, nil
}

// Background processes

// runPeriodicReasoning runs periodic reasoning processes
func (cc *CognitiveCoordinator) runPeriodicReasoning() {
	ticker := time.NewTicker(time.Minute * 5)
	defer ticker.Stop()
	
	for {
		select {
		case <-cc.ctx.Done():
			return
		case <-ticker.C:
			cc.performPeriodicReasoning()
		}
	}
}

// performPeriodicReasoning performs periodic reasoning maintenance
func (cc *CognitiveCoordinator) performPeriodicReasoning() {
	cc.mutex.RLock()
	collections := make([]*CognitiveCollection, 0)
	for _, collection := range cc.collections {
		if collection.ReasoningConfig.EnableForwardChaining {
			collections = append(collections, collection)
		}
	}
	cc.mutex.RUnlock()
	
	for _, collection := range collections {
		plnReasoner := NewPLNReasoner(cc.reasoningEngine)
		newAtoms, err := plnReasoner.ForwardChain(cc.ctx, collection.ReasoningConfig.MaxInferenceDepth)
		if err != nil {
			cc.logger.Error("Failed to perform forward chaining",
				zap.String("collection_id", collection.ID),
				zap.Error(err))
			continue
		}
		
		if len(newAtoms) > 0 {
			cc.logger.Debug("Periodic reasoning completed",
				zap.String("collection_id", collection.ID),
				zap.Int("new_atoms", len(newAtoms)))
		}
	}
}

// runAttentionMaintenance runs attention maintenance processes
func (cc *CognitiveCoordinator) runAttentionMaintenance() {
	ticker := time.NewTicker(time.Minute * 2)
	defer ticker.Stop()
	
	for {
		select {
		case <-cc.ctx.Done():
			return
		case <-ticker.C:
			cc.performAttentionMaintenance()
		}
	}
}

// performAttentionMaintenance performs attention value maintenance
func (cc *CognitiveCoordinator) performAttentionMaintenance() {
	// Decay attention values for all atoms
	cc.atomSpace.mutex.RLock()
	atomIDs := make([]string, 0, len(cc.atomSpace.atoms))
	for atomID := range cc.atomSpace.atoms {
		atomIDs = append(atomIDs, atomID)
	}
	cc.atomSpace.mutex.RUnlock()
	
	for _, atomID := range atomIDs {
		err := cc.attentionAllocation.UpdateAttention(atomID, AttentionDecayRate) // Small decay
		if err != nil {
			// Ignore errors for attention updates
			continue
		}
	}
}

// GetAtomSpace returns the AtomSpace instance
func (cc *CognitiveCoordinator) GetAtomSpace() *AtomSpace {
	return cc.atomSpace
}

// GetPatternMatcher returns the PatternMatcher instance
func (cc *CognitiveCoordinator) GetPatternMatcher() *PatternMatcher {
	return cc.patternMatcher
}

// GetAttentionAllocation returns the AttentionAllocation instance
func (cc *CognitiveCoordinator) GetAttentionAllocation() *AttentionAllocation {
	return cc.attentionAllocation
}

// GetCollections returns all collections
func (cc *CognitiveCoordinator) GetCollections() map[string]*CognitiveCollection {
	cc.mutex.RLock()
	defer cc.mutex.RUnlock()
	
	result := make(map[string]*CognitiveCollection)
	for id, collection := range cc.collections {
		result[id] = collection
	}
	return result
}

// GetCollectionByName returns a collection by name
func (cc *CognitiveCoordinator) GetCollectionByName(name string) *CognitiveCollection {
	cc.mutex.RLock()
	defer cc.mutex.RUnlock()
	
	for _, collection := range cc.collections {
		if collection.Name == name {
			return collection
		}
	}
	return nil
}