# Runtime Diagrams

Last reviewed: 2026-03-18  
Scope: interactive runtime path only

## 1) Global Interactive KG Flow

```mermaid
flowchart TD
    cliMain["main_kg.py main()"] --> infraCheck["_check_infrastructure()"]
    infraCheck --> resolveGoal["_resolve_goal(args)"]
    resolveGoal --> runLoop["run_interactive_kg_session() loop"]
    runLoop --> sessionOrch["session_orchestrator()"]

    sessionOrch --> bootstrapStream["bootstrap_graph_with_memory.stream(...)"]
    bootstrapStream --> interruptCheck{"interrupt emitted?"}
    interruptCheck -->|yes| returnInterrupted["return INTERRUPTED response"]
    returnInterrupted --> userReply["CLI prompt user reply"]
    userReply --> runLoop

    interruptCheck -->|no| contractCheck{"bootstrap_contract present?"}
    contractCheck -->|no| failBootstrap["FAILURE_BOOTSTRAP_REQUIRED"]
    contractCheck -->|yes| seedAwg["seed_awg_from_bootstrap()"]

    seedAwg --> mainLoop{"decision == CONTINUE and iter < max?"}
    mainLoop -->|yes| runInner["run inner_loop per focus concept"]
    runInner --> consolidate["awg_consolidator()"]
    consolidate --> criteria["criteria_check()"]
    criteria --> mainLoop
    mainLoop -->|no| orderNodes["dfs_postorder() over final AWG"]
    orderNodes --> contentGate{"enable content and ordered nodes?"}
    contentGate -->|yes| contentBatch["generate educational content batches"]
    contentGate -->|no| finalize["build session summary"]
    contentBatch --> finalize
    finalize --> returnFinal["return final summary"]
```

## 2) Bootstrap State Machine

```mermaid
flowchart TD
    startNode["START"] --> extractNode["bootstrap_extract"]
    extractNode --> routeExtract{"route_after_bootstrap_extract"}

    routeExtract -->|missing fields and rounds left| askNode["bootstrap_ask (interrupt)"]
    askNode --> extractNode

    routeExtract -->|round limit or default| proceedNode["bootstrap_proceed_gate (interrupt)"]
    proceedNode --> routeProceed{"route_after_bootstrap_proceed_gate"}

    routeProceed -->|proceed requested| finalizeNode["bootstrap_finalize_contract"]
    routeProceed -->|max rounds reached| proceedNode
    routeProceed -->|more details path| askNode

    routeExtract -->|proceed requested| finalizeNode
    finalizeNode --> endNode["END"]
```

## 3) Concept Research Graph (Per Focus Concept)

```mermaid
flowchart TD
    startCR["START"] --> initialProfile["initial_profile_research"]
    initialProfile --> routeAction1{"route_after_action"}
    routeAction1 --> webSearch1["web_search"]
    routeAction1 --> contentExtract1["content_extractor"]
    routeAction1 --> collect1["collect_research"]
    webSearch1 --> collect1
    contentExtract1 --> collect1

    collect1 --> routeResearch1{"route_after_research"}
    routeResearch1 --> proposeProfile["propose_profile"]
    proposeProfile --> evalProfile["evaluate_profile"]
    evalProfile --> profileDone{"profile_completed"}
    profileDone -->|incomplete| actionProfile["action_profile"]
    actionProfile --> routeAction1
    profileDone -->|complete| relatedNode["get_related_concepts"]

    relatedNode --> routeRelated{"route_after_related"}
    routeRelated --> inferRel["infer_relationship"]
    routeRelated --> mergeRelated["merge_related_concepts"]
    inferRel --> mergeRelated

    mergeRelated --> personalizationRouter{"route_after_personalization_router"}
    personalizationRouter -->|no request| initPrereq["initial_prerequisite_research"]
    personalizationRouter -->|has request| pPre["personalization_preprocess"]
    pPre --> pFit["personalization_fit"]
    pFit --> pMode["personalization_mode"]
    pMode --> pDelivery["personalization_delivery"]
    pDelivery --> pAssess["personalization_assessment"]
    pAssess --> pPolicy["personalization_prereq_policy"]

    pPolicy --> policyRoute{"route_after_personalization_prereq_policy"}
    policyRoute -->|stop| mergePrereq["merge_prerequisites"]
    policyRoute -->|expand/limit| initPrereq

    initPrereq --> routeAction2{"route_after_action"}
    routeAction2 --> webSearch2["web_search"]
    routeAction2 --> contentExtract2["content_extractor"]
    routeAction2 --> collect2["collect_research"]
    webSearch2 --> collect2
    contentExtract2 --> collect2

    collect2 --> routeResearch2{"route_after_research"}
    routeResearch2 --> proposePrereq["propose_prerequisites"]
    proposePrereq --> evalPrereq["evaluate_prerequisites"]
    evalPrereq --> prereqDone{"prerequisites_completed"}
    prereqDone -->|incomplete| actionPrereq["action_prerequisites"]
    actionPrereq --> routeAction2
    prereqDone -->|complete| mergePrereq

    mergePrereq --> endCR["END"]
```

## 4) Commit and Resume Sequence

```mermaid
sequenceDiagram
    participant User as UserCLI
    participant Main as main_kg.py
    participant Orch as session_orchestrator
    participant Boot as bootstrap_graph_with_memory
    participant KG4 as awg_consolidator
    participant Content as content_generator
    participant PKG as PKGInterface
    participant Neo4j as Neo4jDB
    participant ER as EducationalReportsRepository

    Main->>Orch: initial request(goal_string, thread_id)
    Orch->>Boot: stream(BootstrapState, configurable.thread_id)
    Boot-->>Orch: interrupt(question)
    Orch-->>Main: status=INTERRUPTED
    Main->>User: show question, collect response
    Main->>Orch: resume request(interrupt_feedback, same thread_id)
    Orch->>Boot: stream(Command(resume=feedback), same thread_id)
    Boot-->>Orch: final state with bootstrap_contract

    Orch->>KG4: consolidate inner loop outputs
    KG4->>PKG: commit_changes(nodes, edges, deletes)
    PKG->>Neo4j: create/update node and relationship records
    Neo4j-->>PKG: commit results / rejections
    PKG-->>KG4: committed + rejected + errors
    KG4-->>Orch: updated_awg + consolidation_status
    Orch->>Orch: dfs_postorder() for learning progression
    Orch->>Content: generate per-concept educational reports
    Content->>ER: create_report(...) for each concept
    ER-->>Content: report IDs / persistence status
    Content-->>Orch: success/failure results
    Orch-->>Main: final session summary
```

## Related Modules

- [Runtime Index Home](../README.md)
- [Entry Point and Interactive Loop](../01-entrypoint-and-interactive-loop.md)
- [Bootstrap State Machine](../02-bootstrap-state-machine.md)
- [Commit Paths and Checkpointing](../07-commit-paths-neo4j-and-session-checkpointing.md)
- [Post-Expansion Ordering and Content Generation](../10-post-expansion-ordering-and-content-generation.md)
