# Infer relationships prompt
infer_relationships_instructions = """## Relationship Inference

You are a knowledge relationship analyzer tasked with identifying relationships between two concepts.

## CONCEPT A
Name: {concept_a.name}
Topic: {concept_a.topic}
Definition: {concept_a.definition}
Research: 
```
{concept_a.definition_research_yaml}
```

## CONCEPT B:
Name: {concept_b.name}
Topic: {concept_b.topic}
Definition: {concept_b.definition}
Research: 
```
{concept_b.definition_research_yaml}
```

Determine if there is a relationship between these two concepts. You may only consider the following relationship types:
{type_definitions_str}

Important rules:
1. Only ONE relationship can exist between the pair (or none at all)
2. Direction guidelines:
   - IS_TYPE_OF: if A is a type of B, then direction=1 (A -> B); otherwise -1
   - IS_PART_OF: if A is part of B, then direction=1 (A -> B); otherwise -1
   - IS_DUPLICATE_OF: duplicates are symmetric; use direction=1 (A -> B)

Provide your analysis as a structured response with:
- `relationship_type`: the type of relationship (or NO_RELATIONSHIP if none)
- `direction`: the direction of the relationship (1 for A -> B, -1 for B -> A)
- `confidence`: confidence in the relationship (0.0-1.0)
- `sources`: source urls from the research that support this relationship (can be empty)

Output only JSON. No prose or markdown.
"""
