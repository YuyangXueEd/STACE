# Core Workflows

The following sequence diagrams illustrate critical system workflows, showing component interactions and data flow.

## Inner Research Loop Workflow

```mermaid
sequenceDiagram
    participant User
    participant ILO as Inner Loop Orchestrator
    participant HG as Hypothesis Generator
    participant Critic
    participant QG as Query Generator
    participant RAG
    participant EE as Experiment Executor
    participant K8s as Kubernetes GPU Job
    participant Eval as Evaluator
    participant LSR as Loop State Repository

    User->>ILO: start_task(task_id, task_type, config)
    ILO->>LSR: save_state(INITIALIZED)

    loop Until vulnerability found OR max iterations
        ILO->>LSR: update_state(HYPOTHESIS_GENERATION)
        ILO->>HG: generate_hypothesis(context)
        HG->>RAG: search(previous_queries) [for context]
        RAG-->>HG: retrieved_papers
        HG->>Memory: retrieve_similar(task_type, prev_hypothesis)
        Memory-->>HG: past_successes
        HG-->>ILO: hypothesis

        alt iteration > 1
            ILO->>LSR: update_state(CRITIC_DEBATE)
            ILO->>Critic: critique_hypothesis(hypothesis, past_results)
            Critic-->>ILO: critic_feedback
            ILO->>HG: refine_hypothesis(hypothesis, critic_feedback)
            HG-->>ILO: refined_hypothesis
        end

        ILO->>LSR: update_state(RAG_RETRIEVAL)
        ILO->>QG: generate_queries(hypothesis, feedback)
        QG-->>ILO: queries
        ILO->>RAG: search(queries, top_k=5)
        RAG-->>ILO: retrieved_papers

        ILO->>LSR: update_state(EXPERIMENT_EXECUTION)
        ILO->>EE: execute_experiment(hypothesis, task_type)
        EE->>K8s: submit_gpu_job(tool, params)
        K8s-->>EE: job_id

        loop Poll until complete (timeout 30min)
            EE->>K8s: poll_job_status(job_id)
            K8s-->>EE: status
        end

        K8s-->>EE: experiment_results
        EE-->>ILO: experiment_result

        ILO->>LSR: update_state(EVALUATION)
        ILO->>Eval: evaluate_results(experiment_result, hypothesis, task_type)

        alt VLM-based evaluation (concept-erasure)
            Eval->>OpenRouter: VLM call (image leakage detection)
            OpenRouter-->>Eval: vlm_response
        end

        Eval-->>ILO: evaluation

        ILO->>LSR: update_state(FEEDBACK)
        ILO->>LSR: save_iteration_result(iteration_result)

        alt vulnerability_detected == True
            ILO->>Memory: store_success(memory_entry)
            ILO->>LSR: update_state(COMPLETED, vulnerability_found=True)
        else max_iterations reached
            ILO->>LSR: update_state(COMPLETED, vulnerability_found=False)
        else continue
            ILO->>LSR: update_state(HYPOTHESIS_GENERATION)
        end
    end

    ILO->>ATR: create_trace(task_id, loop_state, iterations)
    ATR-->>ILO: attack_trace
    ILO-->>User: task_completed(attack_trace)
```

## Outer Reporting Loop Workflow

```mermaid
sequenceDiagram
    participant ILO as Inner Loop Orchestrator
    participant OLO as Outer Loop Orchestrator
    participant Reporter
    participant RAG
    participant Judges as Judge Agents (x3-5)
    participant RR as Report Repository
    participant PV as Persistent Volumes

    ILO->>OLO: start_reporting(task_id, attack_trace)

    OLO->>Reporter: generate_report(attack_trace, retrieved_papers)
    Reporter->>RAG: get_paper_metadata(paper_ids) [for citations]
    RAG-->>Reporter: citation_metadata
    Reporter->>OpenRouter: LLM call (generate Introduction)
    OpenRouter-->>Reporter: introduction_text
    Reporter->>OpenRouter: LLM call (generate Methods)
    OpenRouter-->>Reporter: methods_text
    Reporter->>OpenRouter: LLM call (generate Experiments)
    OpenRouter-->>Reporter: experiments_text
    Reporter->>OpenRouter: LLM call (generate Results)
    OpenRouter-->>Reporter: results_text
    Reporter->>OpenRouter: LLM call (generate Discussion)
    OpenRouter-->>Reporter: discussion_text
    Reporter->>OpenRouter: LLM call (generate Conclusion)
    OpenRouter-->>Reporter: conclusion_text
    Reporter-->>OLO: report

    OLO->>RR: save_report(report)
    RR->>PV: write(report.md, report.json)

    par Concurrent Judge Evaluations
        OLO->>Judges: evaluate_report(report, "Security Expert")
        Judges->>OpenRouter: LLM call (Security Expert persona)
        OpenRouter-->>Judges: evaluation_1

        OLO->>Judges: evaluate_report(report, "ML Researcher")
        Judges->>OpenRouter: LLM call (ML Researcher persona)
        OpenRouter-->>Judges: evaluation_2

        OLO->>Judges: evaluate_report(report, "Privacy Advocate")
        Judges->>OpenRouter: LLM call (Privacy Advocate persona)
        OpenRouter-->>Judges: evaluation_3

        OLO->>Judges: evaluate_report(report, "Skeptical Reviewer")
        Judges->>OpenRouter: LLM call (Skeptical Reviewer persona)
        OpenRouter-->>Judges: evaluation_4

        OLO->>Judges: evaluate_report(report, "Industry Practitioner")
        Judges->>OpenRouter: LLM call (Industry Practitioner persona)
        OpenRouter-->>Judges: evaluation_5
    end

    Judges-->>OLO: judge_evaluations (all)
    OLO->>RR: save_judge_evaluations(evaluations)
    RR->>PV: write(judges/*.json)

    OLO-->>ILO: reporting_completed(report, evaluations)
```

## RAG Indexing and Retrieval Workflow

```mermaid
sequenceDiagram
    participant Setup as Setup Script
    participant RAG as RAG System
    participant FAISS as FAISS Index
    participant Embedder as Sentence-Transformers
    participant Papers as Paper Corpus (PDFs)

    Note over Setup,Papers: One-time Indexing (Setup Phase)

    Setup->>RAG: index_papers("rag/papers/")
    RAG->>Papers: list_pdfs(directory)
    Papers-->>RAG: pdf_files

    loop For each PDF
        RAG->>PyMuPDF: extract_text(pdf_file)
        PyMuPDF-->>RAG: full_text
        RAG->>RAG: chunk_text(full_text, chunk_size=512, overlap=50)
        RAG->>Embedder: encode(chunks)
        Embedder-->>RAG: embeddings
        RAG->>FAISS: add_vectors(embeddings, metadata)
    end

    RAG->>FAISS: save_index("rag/faiss_index.bin")
    FAISS-->>RAG: index_saved

    Note over Setup,Papers: Runtime Retrieval (Inner Loop)

    participant QG as Query Generator
    QG->>RAG: search(query="membership inference attacks", top_k=5)
    RAG->>Embedder: encode(query)
    Embedder-->>RAG: query_embedding
    RAG->>FAISS: search(query_embedding, k=5)
    FAISS-->>RAG: top_k_results (scores, chunk_ids)
    RAG->>RAG: retrieve_chunks_and_metadata(chunk_ids)
    RAG-->>QG: retrieved_papers (with chunks, scores, citations)
```
