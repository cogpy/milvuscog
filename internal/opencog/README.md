# OpenCog Integration with Milvus

This package provides a cognitive framework layer on top of Milvus's distributed vector database, implementing OpenCog's cognitive architecture principles for enhanced AI capabilities.

## Overview

The OpenCog integration extends Milvus with:

1. **AtomSpace**: Knowledge representation using atoms, nodes, and links
2. **Cognitive Reasoning**: PLN (Probabilistic Logic Networks) inference engine
3. **Pattern Matching**: Advanced pattern recognition and matching
4. **Attention Allocation**: Dynamic attention management for cognitive focus
5. **Cognitive Querying**: Intent-aware vector search with reasoning

## Core Components

### AtomSpace (`atomspace.go`)

The central knowledge representation structure that stores atoms representing concepts, relations, and vectors with cognitive properties.

```go
// Create an AtomSpace
atomSpace := opencog.NewAtomSpace()

// Create a vector atom
vectorAtom := atomSpace.CreateVectorAtom(
    "vector_1", 
    "DocumentVector", 
    []float32{0.1, 0.2, 0.3, 0.4},
    &opencog.TruthValue{Strength: 0.9, Confidence: 0.8},
)

// Add to AtomSpace
err := atomSpace.AddAtom(vectorAtom)
```

### Reasoning Engine (`reasoning.go`)

Implements Probabilistic Logic Networks (PLN) for forward and backward chaining inference.

```go
reasoningEngine := opencog.NewReasoningEngine(atomSpace)
plnReasoner := opencog.NewPLNReasoner(reasoningEngine)

// Forward chaining inference
newAtoms, err := plnReasoner.ForwardChain(ctx, maxSteps)

// Backward chaining with goal
proof, err := plnReasoner.BackwardChain(ctx, goal, maxDepth)
```

### Cognitive Coordinator (`coordinator.go`)

Manages cognitive collections and integrates with Milvus's distributed architecture.

```go
coordinator := opencog.NewCognitiveCoordinator()
err := coordinator.Start()

// Create cognitive collection
schema := &opencog.CognitiveSchema{
    VectorField: &opencog.CognitiveField{
        Name:      "embedding",
        Type:      opencog.VectorFieldType,
        Dimension: 768,
        AtomType:  opencog.VectorNode,
    },
    ConceptFields: []*opencog.CognitiveField{
        {
            Name:     "category",
            Type:     opencog.ConceptFieldType,
            AtomType: opencog.ConceptNode,
        },
    },
}

collection, err := coordinator.CreateCognitiveCollection(ctx, "documents", schema, nil)
```

### Cognitive Proxy (`proxy.go`)

Provides cognitive query processing with intent understanding and reasoning.

```go
proxy := opencog.NewCognitiveProxy(coordinator)

// Cognitive query with reasoning
req := &opencog.CognitiveQueryRequest{
    CollectionName:   "documents",
    QueryVector:      queryEmbedding,
    TopK:             10,
    ConceptFilters:   []string{"technology", "AI"},
    ReasoningEnabled: true,
    IntentQuery:      "find documents about machine learning",
}

response, err := proxy.ProcessCognitiveQuery(ctx, req)
```

## Key Features

### 1. Cognitive Vector Representation

Vectors are represented as cognitive atoms with:
- **Truth Values**: Strength and confidence measures
- **Attention Values**: Short/long-term importance and focus
- **Concept Links**: Semantic associations with concepts
- **Relation Links**: Connections to other vectors/concepts

### 2. Intelligent Reasoning

- **Forward Chaining**: Automatic inference of new knowledge
- **Backward Chaining**: Goal-directed reasoning
- **Pattern Matching**: Sophisticated pattern recognition
- **Truth Value Propagation**: Probabilistic reasoning with uncertainty

### 3. Attention-Based Focus

- **Dynamic Attention**: Automatic attention allocation based on usage
- **Focus Management**: Concentrated processing on important atoms
- **Attention Decay**: Natural forgetting of unused information
- **Attention Spreading**: Neighborhood activation patterns

### 4. Intent-Aware Querying

- **Natural Language Processing**: Intent extraction from queries
- **Concept Expansion**: Automatic concept inference and expansion
- **Contextual Filtering**: Context-aware result ranking
- **Reasoning-Enhanced Search**: Results augmented with inferred knowledge

## Architecture Integration

The OpenCog layer integrates seamlessly with Milvus's distributed architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Cognitive Layer                          │
├─────────────────────────────────────────────────────────────┤
│  CognitiveProxy  │  AtomSpace  │  ReasoningEngine  │  PLN   │
├─────────────────────────────────────────────────────────────┤
│              CognitiveCoordinator                           │
├─────────────────────────────────────────────────────────────┤
│                    Milvus Core                              │
├─────────────────────────────────────────────────────────────┤
│  Proxy  │  QueryCoord  │  DataCoord  │  IndexCoord  │  Root  │
├─────────────────────────────────────────────────────────────┤
│              QueryNode  │  DataNode  │  IndexNode           │
├─────────────────────────────────────────────────────────────┤
│                  Vector Storage Layer                       │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Basic Vector Operations

```go
// Initialize cognitive system
coordinator := opencog.NewCognitiveCoordinator()
coordinator.Start()
defer coordinator.Stop()

// Create collection
collection, err := coordinator.CreateCognitiveCollection(ctx, "knowledge_base", schema, nil)

// Insert cognitive vector
vectorID, err := coordinator.InsertCognitiveVector(
    ctx,
    collection.ID,
    embedding,
    []string{"AI", "machine_learning", "neural_networks"},
    map[string]string{"similar_to": "related_concept_id"},
    &opencog.TruthValue{Strength: 0.95, Confidence: 0.9},
)

// Perform cognitive search
result, err := coordinator.CognitiveSearch(
    ctx,
    collection.ID,
    queryVector,
    topK,
    []string{"AI"}, // concept filters
    true, // enable reasoning
)
```

### Advanced Reasoning

```go
// Pattern matching
pattern := &opencog.Atom{
    Type: opencog.InheritanceLink,
    OutgoingSet: []*opencog.Atom{
        {Type: opencog.ConceptNode, Name: "variable_x"},
        {Type: opencog.ConceptNode, Name: "AI"},
    },
}

matches, err := coordinator.patternMatcher.MatchPattern(pattern, constraints)

// Inference reasoning
req := &opencog.CognitiveReasoningRequest{
    CollectionName: "knowledge_base",
    Mode:          "forward",
    MaxSteps:      5,
}

response, err := proxy.ProcessCognitiveReasoning(ctx, req)
```

### Intent-Based Querying

```go
// Natural language cognitive query
req := &opencog.CognitiveQueryRequest{
    CollectionName:   "documents",
    QueryVector:      embedQuery("deep learning research papers"),
    TopK:             20,
    ReasoningEnabled: true,
    IntentQuery:      "find recent papers on transformer architectures for NLP",
    ContextFilters: map[string]interface{}{
        "year": []int{2023, 2024},
        "domain": "NLP",
    },
    AttentionFocus: []string{"transformer", "attention_mechanism"},
}

response, err := proxy.ProcessCognitiveQuery(ctx, req)

// Response includes:
// - Ranked vector results
// - Inferred concepts
// - Reasoning steps
// - Attention mappings
```

## Configuration

### Reasoning Configuration

```go
config := &opencog.ReasoningConfig{
    EnableForwardChaining:  true,
    EnableBackwardChaining: true,
    MaxInferenceDepth:     5,
    InferenceInterval:     time.Minute * 5,
    AttentionThreshold:    50,
    TruthValueThreshold:   0.7,
}
```

### Cognitive Schema Definition

```go
schema := &opencog.CognitiveSchema{
    VectorField: &opencog.CognitiveField{
        Name:        "embedding",
        Type:        opencog.VectorFieldType,
        Dimension:   768,
        AtomType:    opencog.VectorNode,
        Constraints: map[string]interface{}{
            "normalize": true,
            "index_type": "HNSW",
        },
    },
    ConceptFields: []*opencog.CognitiveField{
        {
            Name:     "category",
            Type:     opencog.ConceptFieldType,
            AtomType: opencog.ConceptNode,
        },
        {
            Name:     "keywords",
            Type:     opencog.ConceptFieldType,
            AtomType: opencog.ConceptNode,
        },
    },
    RelationFields: []*opencog.CognitiveField{
        {
            Name:     "related_to",
            Type:     opencog.RelationFieldType,
            AtomType: opencog.SimilarityLink,
        },
    },
    TruthValueField: &opencog.CognitiveField{
        Name: "confidence",
        Type: opencog.TruthValueFieldType,
    },
}
```

## Performance Considerations

1. **AtomSpace Size**: Monitor atom count and implement cleanup strategies
2. **Reasoning Complexity**: Limit inference depth and steps for real-time queries
3. **Attention Management**: Regular attention decay prevents memory bloat
4. **Distributed Processing**: Leverage Milvus's distributed architecture for scale
5. **Caching**: Cache frequent patterns and reasoning results

## Testing

Run the test suite:

```bash
cd internal/opencog
go test -v ./...
```

Tests cover:
- AtomSpace operations
- Reasoning engine functionality
- Pattern matching
- Attention allocation
- Cognitive coordinator
- Proxy integration

## Future Enhancements

1. **Enhanced NLP**: Advanced natural language processing for intent understanding
2. **Temporal Reasoning**: Time-aware cognitive processing
3. **Distributed AtomSpace**: Scale AtomSpace across multiple nodes
4. **Learning Algorithms**: Online learning and adaptation
5. **Visualization**: Cognitive graph visualization tools
6. **Integration APIs**: REST/GraphQL APIs for external integration

## References

- [OpenCog Framework](https://opencog.org/)
- [Probabilistic Logic Networks](https://wiki.opencog.org/w/PLN)
- [AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
- [Milvus Documentation](https://milvus.io/docs)