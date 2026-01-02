# Infer relationships prompt
infer_relationships_instructions = """## Relationship Inference

You are a knowledge relationship analyzer tasked with identifying relationships between two concepts A and B given their profiles:
<A>
{concept_a_str}
</A>

<B>
{concept_b_str}
</B>

Determine if there is a relationship between these two concepts. You may only consider the following relationship types:
<relationship_types>
{types_str}
</relationship_types>

Instructions:
1. Only ONE relationship can exist between the pair (or none at all)
2. Direction guidelines:
   - IS_TYPE_OF: if A is a type of B, then direction=1 (A -> B); otherwise -1
   - IS_PART_OF: if A is part of B, then direction=1 (A -> B); otherwise -1
   - IS_DUPLICATE_OF: duplicates are symmetric; use direction=1 (A -> B)

Output your analysis as a structured response with:
- `relationship_type`: the type of relationship (or NO_RELATIONSHIP if none)
- `direction`: the direction of the relationship (1 for A -> B, -1 for B -> A)
- `confidence`: strength/confidence in the relationship (0.0-1.0)
- `sources`: source urls from the profiles of A and B that support this relationship (can be empty)

Output only JSON. No prose or markdown.
"""
