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

	"github.com/milvus-io/milvus/pkg/v2/log"
	"go.uber.org/zap"
)

// ReasoningEngine provides cognitive reasoning capabilities over the AtomSpace
type ReasoningEngine struct {
	atomSpace *AtomSpace
	logger    *zap.Logger
}

// NewReasoningEngine creates a new reasoning engine
func NewReasoningEngine(atomSpace *AtomSpace) *ReasoningEngine {
	return &ReasoningEngine{
		atomSpace: atomSpace,
		logger:    log.L(),
	}
}

// InferenceRule represents a cognitive inference rule
type InferenceRule struct {
	Name        string
	Precondition func([]*Atom) bool
	Consequence  func([]*Atom) []*Atom
	Strength    float32
}

// PLNReasoner implements Probabilistic Logic Networks reasoning
type PLNReasoner struct {
	engine *ReasoningEngine
	rules  []*InferenceRule
}

// NewPLNReasoner creates a new PLN reasoner
func NewPLNReasoner(engine *ReasoningEngine) *PLNReasoner {
	pln := &PLNReasoner{
		engine: engine,
		rules:  make([]*InferenceRule, 0),
	}
	pln.initializeBasicRules()
	return pln
}

// initializeBasicRules sets up basic inference rules
func (pln *PLNReasoner) initializeBasicRules() {
	// Deduction rule: If A->B and B->C, then A->C
	deductionRule := &InferenceRule{
		Name: "Deduction",
		Precondition: func(atoms []*Atom) bool {
			return len(atoms) >= 2 && 
				atoms[0].Type == InheritanceLink && 
				atoms[1].Type == InheritanceLink &&
				len(atoms[0].OutgoingSet) >= 2 &&
				len(atoms[1].OutgoingSet) >= 2 &&
				atoms[0].OutgoingSet[1].ID == atoms[1].OutgoingSet[0].ID
		},
		Consequence: func(atoms []*Atom) []*Atom {
			// Create new inheritance link A->C
			return []*Atom{
				pln.engine.atomSpace.createInheritanceLink(
					atoms[0].OutgoingSet[0],
					atoms[1].OutgoingSet[1],
					pln.calculateDeductionStrength(atoms[0], atoms[1]),
				),
			}
		},
		Strength: 0.9,
	}

	// Similarity rule: If A is similar to B and B is similar to C, then A is similar to C
	similarityRule := &InferenceRule{
		Name: "Similarity",
		Precondition: func(atoms []*Atom) bool {
			return len(atoms) >= 2 && 
				atoms[0].Type == SimilarityLink && 
				atoms[1].Type == SimilarityLink &&
				len(atoms[0].OutgoingSet) >= 2 &&
				len(atoms[1].OutgoingSet) >= 2 &&
				(atoms[0].OutgoingSet[1].ID == atoms[1].OutgoingSet[0].ID ||
				 atoms[0].OutgoingSet[0].ID == atoms[1].OutgoingSet[1].ID)
		},
		Consequence: func(atoms []*Atom) []*Atom {
			var atom1, atom2 *Atom
			if atoms[0].OutgoingSet[1].ID == atoms[1].OutgoingSet[0].ID {
				atom1 = atoms[0].OutgoingSet[0]
				atom2 = atoms[1].OutgoingSet[1]
			} else {
				atom1 = atoms[0].OutgoingSet[1]
				atom2 = atoms[1].OutgoingSet[0]
			}
			
			return []*Atom{
				pln.engine.atomSpace.CreateSimilarityLink(
					fmt.Sprintf("sim_%s_%s", atom1.ID, atom2.ID),
					atom1,
					atom2,
					pln.calculateSimilarityStrength(atoms[0], atoms[1]),
				),
			}
		},
		Strength: 0.8,
	}

	pln.rules = append(pln.rules, deductionRule, similarityRule)
}

// calculateDeductionStrength calculates strength for deduction inference
func (pln *PLNReasoner) calculateDeductionStrength(link1, link2 *Atom) float32 {
	if link1.TruthValue == nil || link2.TruthValue == nil {
		return 0.5
	}
	
	// PLN deduction formula: simplified version
	s1, c1 := link1.TruthValue.Strength, link1.TruthValue.Confidence
	s2, c2 := link2.TruthValue.Strength, link2.TruthValue.Confidence
	
	strength := s1 * s2
	confidence := c1 * c2 * s1 * s2
	
	return strength * confidence
}

// calculateSimilarityStrength calculates strength for similarity inference
func (pln *PLNReasoner) calculateSimilarityStrength(link1, link2 *Atom) float32 {
	if link1.TruthValue == nil || link2.TruthValue == nil {
		return 0.5
	}
	
	s1, c1 := link1.TruthValue.Strength, link1.TruthValue.Confidence
	s2, c2 := link2.TruthValue.Strength, link2.TruthValue.Confidence
	
	// Similarity transitivity with strength degradation
	strength := float32(math.Sqrt(float64(s1 * s2)))
	confidence := float32(math.Min(float64(c1), float64(c2))) * 0.8
	
	return strength * confidence
}

// createInheritanceLink creates a new inheritance link
func (as *AtomSpace) createInheritanceLink(from, to *Atom, strength float32) *Atom {
	return &Atom{
		ID:   fmt.Sprintf("inh_%s_%s", from.ID, to.ID),
		Type: InheritanceLink,
		Name: fmt.Sprintf("inheritance_%s_%s", from.Name, to.Name),
		TruthValue: &TruthValue{
			Strength:   strength,
			Confidence: 0.8,
		},
		AttentionValue: &AttentionValue{
			STI:  75,
			LTI:  0,
			VLTI: false,
		},
		OutgoingSet: []*Atom{from, to},
		IncomingSet: make([]*Atom, 0),
	}
}

// ForwardChain performs forward chaining inference
func (pln *PLNReasoner) ForwardChain(ctx context.Context, maxSteps int) ([]*Atom, error) {
	newAtoms := make([]*Atom, 0)
	
	for step := 0; step < maxSteps; step++ {
		stepAtoms := make([]*Atom, 0)
		
		// Get all atoms for rule matching
		allAtoms := make([]*Atom, 0)
		for _, atom := range pln.engine.atomSpace.atoms {
			allAtoms = append(allAtoms, atom)
		}
		
		// Apply each rule
		for _, rule := range pln.rules {
			// Find combinations of atoms that satisfy the rule
			combinations := pln.findRuleCombinations(allAtoms, rule)
			
			for _, combination := range combinations {
				if rule.Precondition(combination) {
					consequences := rule.Consequence(combination)
					for _, consequence := range consequences {
						// Check if consequence already exists
						if _, exists := pln.engine.atomSpace.GetAtom(consequence.ID); !exists {
							err := pln.engine.atomSpace.AddAtom(consequence)
							if err != nil {
								pln.engine.logger.Error("Failed to add inferred atom", zap.Error(err))
								continue
							}
							stepAtoms = append(stepAtoms, consequence)
							newAtoms = append(newAtoms, consequence)
						}
					}
				}
			}
		}
		
		// If no new atoms were generated, stop
		if len(stepAtoms) == 0 {
			break
		}
		
		pln.engine.logger.Debug("Forward chaining step completed",
			zap.Int("step", step),
			zap.Int("new_atoms", len(stepAtoms)))
	}
	
	return newAtoms, nil
}

// findRuleCombinations finds combinations of atoms that could trigger rules
func (pln *PLNReasoner) findRuleCombinations(atoms []*Atom, rule *InferenceRule) [][]*Atom {
	combinations := make([][]*Atom, 0)
	
	// Simple pairwise combination for basic rules
	for i := 0; i < len(atoms); i++ {
		for j := i + 1; j < len(atoms); j++ {
			combination := []*Atom{atoms[i], atoms[j]}
			combinations = append(combinations, combination)
		}
	}
	
	return combinations
}

// BackwardChain performs goal-directed backward chaining
func (pln *PLNReasoner) BackwardChain(ctx context.Context, goal *Atom, maxDepth int) ([]*Atom, error) {
	return pln.backwardChainRecursive(ctx, goal, maxDepth, make(map[string]bool))
}

// backwardChainRecursive performs recursive backward chaining
func (pln *PLNReasoner) backwardChainRecursive(ctx context.Context, goal *Atom, depth int, visited map[string]bool) ([]*Atom, error) {
	if depth <= 0 || visited[goal.ID] {
		return nil, nil
	}
	
	visited[goal.ID] = true
	proof := make([]*Atom, 0)
	
	// Check if goal already exists in atomspace
	if existingAtom, exists := pln.engine.atomSpace.GetAtom(goal.ID); exists {
		return []*Atom{existingAtom}, nil
	}
	
	// Try to find rules that could produce the goal
	for _, rule := range pln.rules {
		// This is a simplified backward chaining - in practice, 
		// we would need more sophisticated rule matching
		consequences := rule.Consequence([]*Atom{goal})
		for _, consequence := range consequences {
			if consequence.Type == goal.Type {
				// Try to prove the preconditions
				subgoals := pln.extractSubgoals(rule, goal)
				allProved := true
				
				for _, subgoal := range subgoals {
					subproof, err := pln.backwardChainRecursive(ctx, subgoal, depth-1, visited)
					if err != nil || len(subproof) == 0 {
						allProved = false
						break
					}
					proof = append(proof, subproof...)
				}
				
				if allProved {
					proof = append(proof, goal)
					return proof, nil
				}
			}
		}
	}
	
	return nil, fmt.Errorf("cannot prove goal: %s", goal.ID)
}

// extractSubgoals extracts subgoals needed to prove a goal using a specific rule
func (pln *PLNReasoner) extractSubgoals(rule *InferenceRule, goal *Atom) []*Atom {
	// This is a simplified implementation
	// In practice, this would involve more sophisticated pattern matching
	subgoals := make([]*Atom, 0)
	
	if goal.Type == InheritanceLink && len(goal.OutgoingSet) >= 2 {
		// For inheritance links, we might need intermediate concepts
		from := goal.OutgoingSet[0]
		to := goal.OutgoingSet[1]
		
		// Try to find intermediate concept
		intermediates := pln.engine.atomSpace.GetAtomsByType(ConceptNode)
		for _, intermediate := range intermediates {
			if intermediate.ID != from.ID && intermediate.ID != to.ID {
				subgoal1 := pln.engine.atomSpace.createInheritanceLink(from, intermediate, 0.8)
				subgoal2 := pln.engine.atomSpace.createInheritanceLink(intermediate, to, 0.8)
				subgoals = append(subgoals, subgoal1, subgoal2)
				break // Take first intermediate for simplicity
			}
		}
	}
	
	return subgoals
}

// PatternMatcher provides pattern matching capabilities
type PatternMatcher struct {
	atomSpace *AtomSpace
	logger    *zap.Logger
}

// NewPatternMatcher creates a new pattern matcher
func NewPatternMatcher(atomSpace *AtomSpace) *PatternMatcher {
	return &PatternMatcher{
		atomSpace: atomSpace,
		logger:    log.L(),
	}
}

// MatchPattern finds atoms matching a given pattern
func (pm *PatternMatcher) MatchPattern(pattern *Atom, bindingConstraints map[string]func(*Atom) bool) ([]*Atom, error) {
	matches := make([]*Atom, 0)
	
	pm.atomSpace.mutex.RLock()
	defer pm.atomSpace.mutex.RUnlock()
	
	for _, atom := range pm.atomSpace.atoms {
		if pm.matchesPatternWithConstraints(atom, pattern, bindingConstraints) {
			matches = append(matches, atom)
		}
	}
	
	// Sort matches by attention value (STI)
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].AttentionValue == nil || matches[j].AttentionValue == nil {
			return false
		}
		return matches[i].AttentionValue.STI > matches[j].AttentionValue.STI
	})
	
	pm.logger.Debug("Pattern matching completed",
		zap.Int("matches", len(matches)),
		zap.String("pattern_id", pattern.ID))
	
	return matches, nil
}

// matchesPatternWithConstraints checks if an atom matches pattern with constraints
func (pm *PatternMatcher) matchesPatternWithConstraints(atom, pattern *Atom, constraints map[string]func(*Atom) bool) bool {
	// Basic type matching
	if pattern.Type != atom.Type {
		return false
	}
	
	// Name matching (if specified)
	if pattern.Name != "" && pattern.Name != atom.Name {
		return false
	}
	
	// Check binding constraints
	for variable, constraint := range constraints {
		if pattern.Name == variable {
			if !constraint(atom) {
				return false
			}
		}
	}
	
	// Structural matching for links
	if len(pattern.OutgoingSet) > 0 {
		if len(atom.OutgoingSet) != len(pattern.OutgoingSet) {
			return false
		}
		
		for i, patternOutgoing := range pattern.OutgoingSet {
			if !pm.matchesPatternWithConstraints(atom.OutgoingSet[i], patternOutgoing, constraints) {
				return false
			}
		}
	}
	
	return true
}

// AttentionAllocation manages attention allocation across atoms
type AttentionAllocation struct {
	atomSpace *AtomSpace
	logger    *zap.Logger
}

// NewAttentionAllocation creates a new attention allocation manager
func NewAttentionAllocation(atomSpace *AtomSpace) *AttentionAllocation {
	return &AttentionAllocation{
		atomSpace: atomSpace,
		logger:    log.L(),
	}
}

// UpdateAttention updates attention values based on usage and importance
func (aa *AttentionAllocation) UpdateAttention(atomID string, stimulusSTI int16) error {
	atom, exists := aa.atomSpace.GetAtom(atomID)
	if !exists {
		return fmt.Errorf("atom %s not found", atomID)
	}
	
	if atom.AttentionValue == nil {
		atom.AttentionValue = &AttentionValue{STI: 0, LTI: 0, VLTI: false}
	}
	
	// Update STI with decay and stimulus
	oldSTI := atom.AttentionValue.STI
	atom.AttentionValue.STI = int16(float32(oldSTI)*0.9 + float32(stimulusSTI)*0.1)
	
	// Update LTI based on consistent high STI
	if atom.AttentionValue.STI > 150 {
		atom.AttentionValue.LTI++
	} else if atom.AttentionValue.STI < 50 {
		atom.AttentionValue.LTI--
	}
	
	// Set VLTI for very important atoms
	if atom.AttentionValue.LTI > 100 {
		atom.AttentionValue.VLTI = true
	}
	
	aa.logger.Debug("Updated attention values",
		zap.String("atom_id", atomID),
		zap.Int16("old_sti", oldSTI),
		zap.Int16("new_sti", atom.AttentionValue.STI),
		zap.Int16("lti", atom.AttentionValue.LTI))
	
	return nil
}

// FocusAttention focuses attention on specific atoms and their neighborhoods
func (aa *AttentionAllocation) FocusAttention(focusAtoms []string, radius int) error {
	focusSet := make(map[string]bool)
	
	// Add focus atoms
	for _, atomID := range focusAtoms {
		focusSet[atomID] = true
	}
	
	// Expand focus to neighborhood
	for i := 0; i < radius; i++ {
		newFocus := make(map[string]bool)
		for atomID := range focusSet {
			atom, exists := aa.atomSpace.GetAtom(atomID)
			if !exists {
				continue
			}
			
			// Add incoming and outgoing atoms
			for _, incoming := range atom.IncomingSet {
				newFocus[incoming.ID] = true
			}
			for _, outgoing := range atom.OutgoingSet {
				newFocus[outgoing.ID] = true
			}
		}
		
		// Merge new focus atoms
		for atomID := range newFocus {
			focusSet[atomID] = true
		}
	}
	
	// Apply attention boost to focus set
	for atomID := range focusSet {
		err := aa.UpdateAttention(atomID, 50)
		if err != nil {
			aa.logger.Error("Failed to update attention", zap.String("atom_id", atomID), zap.Error(err))
		}
	}
	
	aa.logger.Debug("Focus attention applied",
		zap.Int("focus_atoms", len(focusAtoms)),
		zap.Int("total_focused", len(focusSet)),
		zap.Int("radius", radius))
	
	return nil
}