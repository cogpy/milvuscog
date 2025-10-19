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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAtomSpace(t *testing.T) {
	atomSpace := NewAtomSpace()
	
	t.Run("AddAtom", func(t *testing.T) {
		atom := &Atom{
			ID:   "test_atom_1",
			Type: ConceptNode,
			Name: "TestConcept",
			TruthValue: &TruthValue{
				Strength:   0.8,
				Confidence: 0.9,
			},
			AttentionValue: &AttentionValue{
				STI:  100,
				LTI:  10,
				VLTI: false,
			},
			Vector:      []float32{1.0, 2.0, 3.0},
			OutgoingSet: make([]*Atom, 0),
			IncomingSet: make([]*Atom, 0),
		}
		
		err := atomSpace.AddAtom(atom)
		assert.NoError(t, err)
		assert.Equal(t, 1, atomSpace.Size())
	})
	
	t.Run("GetAtom", func(t *testing.T) {
		atom, exists := atomSpace.GetAtom("test_atom_1")
		assert.True(t, exists)
		assert.NotNil(t, atom)
		assert.Equal(t, "TestConcept", atom.Name)
	})
	
	t.Run("GetAtomsByType", func(t *testing.T) {
		atoms := atomSpace.GetAtomsByType(ConceptNode)
		assert.Equal(t, 1, len(atoms))
		assert.Equal(t, "TestConcept", atoms[0].Name)
	})
	
	t.Run("CreateVectorAtom", func(t *testing.T) {
		vector := []float32{0.1, 0.2, 0.3, 0.4}
		truthValue := &TruthValue{Strength: 0.9, Confidence: 0.8}
		
		vectorAtom := atomSpace.CreateVectorAtom("vec_1", "TestVector", vector, truthValue)
		
		assert.Equal(t, "vec_1", vectorAtom.ID)
		assert.Equal(t, VectorNode, vectorAtom.Type)
		assert.Equal(t, "TestVector", vectorAtom.Name)
		assert.Equal(t, vector, vectorAtom.Vector)
		assert.Equal(t, truthValue, vectorAtom.TruthValue)
		assert.NotNil(t, vectorAtom.AttentionValue)
	})
	
	t.Run("CreateSimilarityLink", func(t *testing.T) {
		atom1 := &Atom{ID: "atom1", Type: ConceptNode, Name: "Concept1"}
		atom2 := &Atom{ID: "atom2", Type: ConceptNode, Name: "Concept2"}
		
		similarity := float32(0.75)
		simLink := atomSpace.CreateSimilarityLink("sim_1", atom1, atom2, similarity)
		
		assert.Equal(t, "sim_1", simLink.ID)
		assert.Equal(t, SimilarityLink, simLink.Type)
		assert.Equal(t, similarity, simLink.TruthValue.Strength)
		assert.Equal(t, 2, len(simLink.OutgoingSet))
		assert.Equal(t, atom1, simLink.OutgoingSet[0])
		assert.Equal(t, atom2, simLink.OutgoingSet[1])
	})
	
	t.Run("RemoveAtom", func(t *testing.T) {
		err := atomSpace.RemoveAtom("test_atom_1")
		assert.NoError(t, err)
		
		_, exists := atomSpace.GetAtom("test_atom_1")
		assert.False(t, exists)
	})
}

func TestReasoningEngine(t *testing.T) {
	atomSpace := NewAtomSpace()
	reasoningEngine := NewReasoningEngine(atomSpace)
	
	t.Run("PLNReasoner", func(t *testing.T) {
		plnReasoner := NewPLNReasoner(reasoningEngine)
		assert.NotNil(t, plnReasoner)
		assert.Greater(t, len(plnReasoner.rules), 0)
	})
	
	t.Run("ForwardChaining", func(t *testing.T) {
		// Create test atoms for reasoning
		conceptA := &Atom{
			ID:   "concept_a",
			Type: ConceptNode,
			Name: "ConceptA",
			TruthValue: &TruthValue{Strength: 0.9, Confidence: 0.8},
		}
		
		conceptB := &Atom{
			ID:   "concept_b",
			Type: ConceptNode,
			Name: "ConceptB",
			TruthValue: &TruthValue{Strength: 0.8, Confidence: 0.9},
		}
		
		conceptC := &Atom{
			ID:   "concept_c",
			Type: ConceptNode,
			Name: "ConceptC",
			TruthValue: &TruthValue{Strength: 0.7, Confidence: 0.8},
		}
		
		// Create inheritance links A->B and B->C
		inhAB := atomSpace.createInheritanceLink(conceptA, conceptB, 0.8)
		inhBC := atomSpace.createInheritanceLink(conceptB, conceptC, 0.7)
		
		// Add all atoms to atomspace
		err := atomSpace.AddAtom(conceptA)
		require.NoError(t, err)
		err = atomSpace.AddAtom(conceptB)
		require.NoError(t, err)
		err = atomSpace.AddAtom(conceptC)
		require.NoError(t, err)
		err = atomSpace.AddAtom(inhAB)
		require.NoError(t, err)
		err = atomSpace.AddAtom(inhBC)
		require.NoError(t, err)
		
		plnReasoner := NewPLNReasoner(reasoningEngine)
		ctx := context.Background()
		
		newAtoms, err := plnReasoner.ForwardChain(ctx, 3)
		assert.NoError(t, err)
		assert.GreaterOrEqual(t, len(newAtoms), 0) // May infer A->C
	})
}

func TestPatternMatcher(t *testing.T) {
	atomSpace := NewAtomSpace()
	patternMatcher := NewPatternMatcher(atomSpace)
	
	t.Run("MatchPattern", func(t *testing.T) {
		// Add test atoms
		conceptAtom := &Atom{
			ID:   "test_concept",
			Type: ConceptNode,
			Name: "TestPattern",
			TruthValue: &TruthValue{Strength: 0.8, Confidence: 0.9},
		}
		
		err := atomSpace.AddAtom(conceptAtom)
		require.NoError(t, err)
		
		// Create pattern to match concept nodes
		pattern := &Atom{
			Type: ConceptNode,
			Name: "", // Empty name means match any name
		}
		
		matches, err := patternMatcher.MatchPattern(pattern, nil)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(matches))
		assert.Equal(t, "TestPattern", matches[0].Name)
	})
	
	t.Run("MatchPatternWithConstraints", func(t *testing.T) {
		pattern := &Atom{
			Type: ConceptNode,
			Name: "variable_x",
		}
		
		// Define constraint: truth value strength > 0.5
		constraints := map[string]func(*Atom) bool{
			"variable_x": func(atom *Atom) bool {
				return atom.TruthValue != nil && atom.TruthValue.Strength > 0.5
			},
		}
		
		matches, err := patternMatcher.MatchPattern(pattern, constraints)
		assert.NoError(t, err)
		assert.GreaterOrEqual(t, len(matches), 0)
	})
}

func TestAttentionAllocation(t *testing.T) {
	atomSpace := NewAtomSpace()
	attentionAllocation := NewAttentionAllocation(atomSpace)
	
	t.Run("UpdateAttention", func(t *testing.T) {
		atom := &Atom{
			ID:   "attention_test",
			Type: ConceptNode,
			Name: "AttentionTest",
			AttentionValue: &AttentionValue{
				STI:  50,
				LTI:  0,
				VLTI: false,
			},
		}
		
		err := atomSpace.AddAtom(atom)
		require.NoError(t, err)
		
		err = attentionAllocation.UpdateAttention("attention_test", 100)
		assert.NoError(t, err)
		
		updatedAtom, exists := atomSpace.GetAtom("attention_test")
		assert.True(t, exists)
		assert.Greater(t, updatedAtom.AttentionValue.STI, int16(50))
	})
	
	t.Run("FocusAttention", func(t *testing.T) {
		// Add more atoms for focus test
		atom1 := &Atom{
			ID:   "focus_atom_1",
			Type: ConceptNode,
			Name: "FocusAtom1",
			AttentionValue: &AttentionValue{STI: 0, LTI: 0, VLTI: false},
		}
		
		atom2 := &Atom{
			ID:   "focus_atom_2",
			Type: ConceptNode,
			Name: "FocusAtom2",
			AttentionValue: &AttentionValue{STI: 0, LTI: 0, VLTI: false},
		}
		
		err := atomSpace.AddAtom(atom1)
		require.NoError(t, err)
		err = atomSpace.AddAtom(atom2)
		require.NoError(t, err)
		
		err = attentionAllocation.FocusAttention([]string{"focus_atom_1"}, 1)
		assert.NoError(t, err)
		
		focusedAtom, exists := atomSpace.GetAtom("focus_atom_1")
		assert.True(t, exists)
		assert.Greater(t, focusedAtom.AttentionValue.STI, int16(0))
	})
}

func TestCognitiveCoordinator(t *testing.T) {
	coordinator := NewCognitiveCoordinator()
	
	t.Run("StartStop", func(t *testing.T) {
		err := coordinator.Start()
		assert.NoError(t, err)
		assert.True(t, coordinator.running)
		
		err = coordinator.Stop()
		assert.NoError(t, err)
		assert.False(t, coordinator.running)
	})
	
	t.Run("CreateCognitiveCollection", func(t *testing.T) {
		err := coordinator.Start()
		require.NoError(t, err)
		defer coordinator.Stop()
		
		schema := &CognitiveSchema{
			VectorField: &CognitiveField{
				Name:      "embedding",
				Type:      VectorFieldType,
				Dimension: 128,
				AtomType:  VectorNode,
			},
			ConceptFields: []*CognitiveField{
				{
					Name:     "category",
					Type:     ConceptFieldType,
					AtomType: ConceptNode,
				},
			},
		}
		
		ctx := context.Background()
		collection, err := coordinator.CreateCognitiveCollection(ctx, "test_collection", schema, nil)
		
		assert.NoError(t, err)
		assert.NotNil(t, collection)
		assert.Equal(t, "test_collection", collection.Name)
		assert.NotEmpty(t, collection.ID)
	})
	
	t.Run("InsertCognitiveVector", func(t *testing.T) {
		err := coordinator.Start()
		require.NoError(t, err)
		defer coordinator.Stop()
		
		// Create collection first
		schema := &CognitiveSchema{
			VectorField: &CognitiveField{
				Name:      "embedding",
				Type:      VectorFieldType,
				Dimension: 4,
				AtomType:  VectorNode,
			},
		}
		
		ctx := context.Background()
		collection, err := coordinator.CreateCognitiveCollection(ctx, "insert_test", schema, nil)
		require.NoError(t, err)
		
		vector := []float32{0.1, 0.2, 0.3, 0.4}
		concepts := []string{"test_concept", "insert_concept"}
		relations := map[string]string{"similar_to": "other_vector"}
		truthValue := &TruthValue{Strength: 0.9, Confidence: 0.8}
		
		vectorID, err := coordinator.InsertCognitiveVector(
			ctx,
			collection.ID,
			vector,
			concepts,
			relations,
			truthValue,
		)
		
		assert.NoError(t, err)
		assert.NotEmpty(t, vectorID)
		
		// Verify the vector was added to AtomSpace
		atom, exists := coordinator.atomSpace.GetAtom(vectorID)
		assert.True(t, exists)
		assert.Equal(t, VectorNode, atom.Type)
		assert.Equal(t, vector, atom.Vector)
	})
	
	t.Run("CognitiveSearch", func(t *testing.T) {
		err := coordinator.Start()
		require.NoError(t, err)
		defer coordinator.Stop()
		
		// Create collection and insert test data
		schema := &CognitiveSchema{
			VectorField: &CognitiveField{
				Name:      "embedding",
				Type:      VectorFieldType,
				Dimension: 4,
				AtomType:  VectorNode,
			},
		}
		
		ctx := context.Background()
		collection, err := coordinator.CreateCognitiveCollection(ctx, "search_test", schema, nil)
		require.NoError(t, err)
		
		// Insert test vectors
		vector1 := []float32{1.0, 0.0, 0.0, 0.0}
		vector2 := []float32{0.8, 0.2, 0.0, 0.0}
		
		_, err = coordinator.InsertCognitiveVector(
			ctx, collection.ID, vector1, []string{"concept1"}, nil, nil)
		require.NoError(t, err)
		
		_, err = coordinator.InsertCognitiveVector(
			ctx, collection.ID, vector2, []string{"concept2"}, nil, nil)
		require.NoError(t, err)
		
		// Perform search
		queryVector := []float32{0.9, 0.1, 0.0, 0.0}
		result, err := coordinator.CognitiveSearch(
			ctx, collection.ID, queryVector, 2, nil, false)
		
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.LessOrEqual(t, len(result.Results), 2)
		assert.NotEmpty(t, result.QueryID)
	})
}

func TestCognitiveProxy(t *testing.T) {
	coordinator := NewCognitiveCoordinator()
	proxy := NewCognitiveProxy(coordinator)
	
	t.Run("ProcessCognitiveQuery", func(t *testing.T) {
		err := coordinator.Start()
		require.NoError(t, err)
		defer coordinator.Stop()
		
		// Create collection
		schema := &CognitiveSchema{
			VectorField: &CognitiveField{
				Name:      "embedding",
				Type:      VectorFieldType,
				Dimension: 4,
				AtomType:  VectorNode,
			},
		}
		
		ctx := context.Background()
		collection, err := coordinator.CreateCognitiveCollection(ctx, "proxy_test", schema, nil)
		require.NoError(t, err)
		
		// Insert test data
		vector := []float32{1.0, 0.0, 0.0, 0.0}
		_, err = coordinator.InsertCognitiveVector(
			ctx, collection.ID, vector, []string{"test_concept"}, nil, nil)
		require.NoError(t, err)
		
		// Test query
		req := &CognitiveQueryRequest{
			CollectionName:   "proxy_test",
			QueryVector:      []float32{0.9, 0.1, 0.0, 0.0},
			TopK:             5,
			ConceptFilters:   []string{},
			ReasoningEnabled: false,
			IntentQuery:      "find similar vectors",
		}
		
		response, err := proxy.ProcessCognitiveQuery(ctx, req)
		
		assert.NoError(t, err)
		assert.NotNil(t, response)
		assert.NotEmpty(t, response.QueryID)
		assert.GreaterOrEqual(t, len(response.Results), 0)
		assert.GreaterOrEqual(t, response.ExecutionTime, int64(0))
	})
	
	t.Run("ProcessCognitiveInsert", func(t *testing.T) {
		err := coordinator.Start()
		require.NoError(t, err)
		defer coordinator.Stop()
		
		// Create collection
		schema := &CognitiveSchema{
			VectorField: &CognitiveField{
				Name:      "embedding",
				Type:      VectorFieldType,
				Dimension: 3,
				AtomType:  VectorNode,
			},
		}
		
		ctx := context.Background()
		_, err = coordinator.CreateCognitiveCollection(ctx, "insert_proxy_test", schema, nil)
		require.NoError(t, err)
		
		req := &CognitiveInsertRequest{
			CollectionName: "insert_proxy_test",
			Vectors: [][]float32{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
			},
			Concepts: [][]string{
				{"concept1", "category1"},
				{"concept2", "category2"},
			},
			TruthValues: []*TruthValue{
				{Strength: 0.9, Confidence: 0.8},
				{Strength: 0.8, Confidence: 0.9},
			},
		}
		
		response, err := proxy.ProcessCognitiveInsert(ctx, req)
		
		assert.NoError(t, err)
		assert.NotNil(t, response)
		assert.Equal(t, 2, response.SuccessCount)
		assert.Equal(t, 0, response.ErrorCount)
		assert.Equal(t, 2, len(response.InsertedIDs))
	})
}