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
	"sync"

	"github.com/milvus-io/milvus/pkg/v2/log"
	"go.uber.org/zap"
)

// AtomType represents different types of atoms in the AtomSpace
type AtomType int

const (
	ConceptNode AtomType = iota
	PredicateNode
	ListLink
	EvaluationLink
	InheritanceLink
	SimilarityLink
	VectorNode
	CognitiveCollectionType
)

// Atom represents a basic unit in the AtomSpace
type Atom struct {
	ID          string
	Type        AtomType
	Name        string
	TruthValue  *TruthValue
	AttentionValue *AttentionValue
	Vector      []float32
	OutgoingSet []*Atom
	IncomingSet []*Atom
}

// TruthValue represents the strength and confidence of an atom
type TruthValue struct {
	Strength   float32
	Confidence float32
}

// AttentionValue represents the importance and focus of an atom
type AttentionValue struct {
	STI  int16  // Short-term importance
	LTI  int16  // Long-term importance
	VLTI bool   // Very long-term importance
}

// AtomSpace is the central knowledge representation structure
type AtomSpace struct {
	atoms       map[string]*Atom
	typeIndices map[AtomType]map[string]*Atom
	mutex       sync.RWMutex
	logger      *zap.Logger
}

// NewAtomSpace creates a new AtomSpace instance
func NewAtomSpace() *AtomSpace {
	return &AtomSpace{
		atoms:       make(map[string]*Atom),
		typeIndices: make(map[AtomType]map[string]*Atom),
		logger:      log.L(),
	}
}

// AddAtom adds an atom to the AtomSpace
func (as *AtomSpace) AddAtom(atom *Atom) error {
	if atom == nil {
		return fmt.Errorf("cannot add nil atom")
	}

	as.mutex.Lock()
	defer as.mutex.Unlock()

	// Add to main atom store
	as.atoms[atom.ID] = atom

	// Add to type index
	if as.typeIndices[atom.Type] == nil {
		as.typeIndices[atom.Type] = make(map[string]*Atom)
	}
	as.typeIndices[atom.Type][atom.ID] = atom

	// Update incoming sets for outgoing atoms
	for _, outgoing := range atom.OutgoingSet {
		if outgoing != nil {
			outgoing.IncomingSet = append(outgoing.IncomingSet, atom)
		}
	}

	as.logger.Debug("Added atom to AtomSpace",
		zap.String("id", atom.ID),
		zap.String("type", fmt.Sprintf("%v", atom.Type)),
		zap.String("name", atom.Name))

	return nil
}

// GetAtom retrieves an atom by ID
func (as *AtomSpace) GetAtom(id string) (*Atom, bool) {
	as.mutex.RLock()
	defer as.mutex.RUnlock()

	atom, exists := as.atoms[id]
	return atom, exists
}

// GetAtomsByType retrieves all atoms of a specific type
func (as *AtomSpace) GetAtomsByType(atomType AtomType) []*Atom {
	as.mutex.RLock()
	defer as.mutex.RUnlock()

	typeMap, exists := as.typeIndices[atomType]
	if !exists {
		return nil
	}

	result := make([]*Atom, 0, len(typeMap))
	for _, atom := range typeMap {
		result = append(result, atom)
	}

	return result
}

// RemoveAtom removes an atom from the AtomSpace
func (as *AtomSpace) RemoveAtom(id string) error {
	as.mutex.Lock()
	defer as.mutex.Unlock()

	atom, exists := as.atoms[id]
	if !exists {
		return fmt.Errorf("atom with ID %s not found", id)
	}

	// Remove from incoming sets
	for _, incoming := range atom.IncomingSet {
		for i, outgoing := range incoming.OutgoingSet {
			if outgoing != nil && outgoing.ID == id {
				incoming.OutgoingSet = append(incoming.OutgoingSet[:i], incoming.OutgoingSet[i+1:]...)
				break
			}
		}
	}

	// Remove from outgoing sets
	for _, outgoing := range atom.OutgoingSet {
		if outgoing != nil {
			for i, incoming := range outgoing.IncomingSet {
				if incoming != nil && incoming.ID == id {
					outgoing.IncomingSet = append(outgoing.IncomingSet[:i], outgoing.IncomingSet[i+1:]...)
					break
				}
			}
		}
	}

	// Remove from indices
	delete(as.atoms, id)
	if typeMap, exists := as.typeIndices[atom.Type]; exists {
		delete(typeMap, id)
	}

	as.logger.Debug("Removed atom from AtomSpace", zap.String("id", id))
	return nil
}

// Size returns the number of atoms in the AtomSpace
func (as *AtomSpace) Size() int {
	as.mutex.RLock()
	defer as.mutex.RUnlock()
	return len(as.atoms)
}

// CreateVectorAtom creates an atom representing a vector with cognitive properties
func (as *AtomSpace) CreateVectorAtom(id, name string, vector []float32, truthValue *TruthValue) *Atom {
	if truthValue == nil {
		truthValue = &TruthValue{Strength: 1.0, Confidence: 0.9}
	}

	return &Atom{
		ID:         id,
		Type:       VectorNode,
		Name:       name,
		TruthValue: truthValue,
		AttentionValue: &AttentionValue{
			STI:  100,
			LTI:  0,
			VLTI: false,
		},
		Vector:      vector,
		OutgoingSet: make([]*Atom, 0),
		IncomingSet: make([]*Atom, 0),
	}
}

// CreateSimilarityLink creates a link representing similarity between two atoms
func (as *AtomSpace) CreateSimilarityLink(id string, atom1, atom2 *Atom, similarity float32) *Atom {
	return &Atom{
		ID:   id,
		Type: SimilarityLink,
		Name: fmt.Sprintf("similarity_%s_%s", atom1.ID, atom2.ID),
		TruthValue: &TruthValue{
			Strength:   similarity,
			Confidence: 0.8,
		},
		AttentionValue: &AttentionValue{
			STI:  50,
			LTI:  0,
			VLTI: false,
		},
		OutgoingSet: []*Atom{atom1, atom2},
		IncomingSet: make([]*Atom, 0),
	}
}

// QueryContext represents a cognitive query context
type QueryContext struct {
	Context context.Context
	Filters map[string]interface{}
	Limit   int
	Offset  int
}

// CognitiveQuery performs pattern matching and reasoning over the AtomSpace
func (as *AtomSpace) CognitiveQuery(ctx *QueryContext, pattern *Atom) ([]*Atom, error) {
	as.mutex.RLock()
	defer as.mutex.RUnlock()

	results := make([]*Atom, 0)

	// Simple pattern matching based on atom type and properties
	for _, atom := range as.atoms {
		if as.matchesPattern(atom, pattern) {
			results = append(results, atom)
			if ctx.Limit > 0 && len(results) >= ctx.Limit {
				break
			}
		}
	}

	as.logger.Debug("Cognitive query completed",
		zap.Int("results", len(results)),
		zap.String("pattern_type", fmt.Sprintf("%v", pattern.Type)))

	return results, nil
}

// matchesPattern checks if an atom matches a given pattern
func (as *AtomSpace) matchesPattern(atom, pattern *Atom) bool {
	if pattern.Type != atom.Type {
		return false
	}

	if pattern.Name != "" && pattern.Name != atom.Name {
		return false
	}

	// Additional pattern matching logic can be added here
	return true
}

// GetRelatedAtoms finds atoms related to a given atom through links
func (as *AtomSpace) GetRelatedAtoms(atomID string, linkType AtomType) []*Atom {
	as.mutex.RLock()
	defer as.mutex.RUnlock()

	atom, exists := as.atoms[atomID]
	if !exists {
		return nil
	}

	related := make([]*Atom, 0)

	// Check incoming links
	for _, incoming := range atom.IncomingSet {
		if incoming.Type == linkType {
			for _, outgoing := range incoming.OutgoingSet {
				if outgoing.ID != atomID {
					related = append(related, outgoing)
				}
			}
		}
	}

	return related
}

// GetAllAtoms returns all atoms in the AtomSpace (for iteration)
func (as *AtomSpace) GetAllAtoms() map[string]*Atom {
	as.mutex.RLock()
	defer as.mutex.RUnlock()
	
	result := make(map[string]*Atom)
	for id, atom := range as.atoms {
		result[id] = atom
	}
	return result
}