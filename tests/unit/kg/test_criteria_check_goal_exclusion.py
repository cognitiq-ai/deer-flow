from src.kg.agent_working_graph import AgentWorkingGraph
from src.kg.base_models import ConceptNode, Relationship, RelationshipType
from src.orchestrator.kg import criteria_check
from src.orchestrator.models import SessionLog


def test_criteria_check_never_returns_goal_as_focus():
    goal = ConceptNode(name="Learn SQL", node_type="goal")
    awg = AgentWorkingGraph()
    awg.add_node(goal)

    decision, focus = criteria_check(
        goal_node_current=goal,
        awg_current=awg,
        iteration_main_cycle=0,
        session_log=SessionLog(),
    )

    assert all(node.node_type != "goal" for node in focus)
    assert decision in {
        "STOP_PREREQUISITES_MET",
        "STOP_MAX_ITERATIONS",
        "CONTINUE_RESEARCH",
    }


def test_criteria_check_traces_prereqs_from_goal_fulfilling_concepts():
    goal = ConceptNode(name="Learn SQL", node_type="goal")
    seed = ConceptNode(name="SQL Basics")
    prereq = ConceptNode(name="Relational Algebra")

    awg = AgentWorkingGraph()
    awg.add_node(goal)
    awg.add_node(seed)
    awg.add_node(prereq)
    awg.add_relationship(
        Relationship(
            source_node_id=seed.id,
            target_node_id=goal.id,
            type=RelationshipType.FULFILS_GOAL,
        )
    )
    awg.add_relationship(
        Relationship(
            source_node_id=seed.id,
            target_node_id=prereq.id,
            type=RelationshipType.HAS_PREREQUISITE,
        )
    )

    decision, focus = criteria_check(
        goal_node_current=goal,
        awg_current=awg,
        iteration_main_cycle=0,
        session_log=SessionLog(),
    )

    assert decision == "CONTINUE_RESEARCH"
    assert any(node.id == prereq.id for node in focus)
