**miLLM**

Mechanistic Interpretability LLM Server

**Business Requirements Document**

Version 1.0

January 29, 2026

**Document Control**

  ----------------- -----------------------------------------------------
  **Attribute**     **Value**

  Document Title    miLLM Business Requirements Document

  Version           1.0

  Status            Draft

  Date              January 29, 2026

  Classification    Internal
  ----------------- -----------------------------------------------------

**1. Executive Summary**

miLLM (Mechanistic Interpretability LLM) is a lightweight, OpenAI
API-compatible inference server designed to run local large language
models with integrated Sparse Autoencoder (SAE) steering capabilities.
Unlike existing solutions such as Ollama or vLLM, miLLM enables users to
hook SAEs into models at runtime, allowing real-time manipulation of
model behavior through feature activation adjustments.

The system addresses a gap in the current tooling landscape: the ability
to apply mechanistic interpretability research to practical inference
scenarios. Users can download models directly from Hugging Face, attach
corresponding SAEs (such as those from the Gemma-Scope project), and
adjust feature strengths to influence model outputs without modifying
system prompts or fine-tuning.

miLLM is designed to pair with miStudio, a companion application for SAE
training and feature discovery. Together, they form a complete pipeline
for interpretability research and applied model steering.

**2. Problem Statement**

**2.1 Current State**

Today, users running local LLMs face several limitations when attempting
to apply mechanistic interpretability techniques:

-   Existing inference servers (Ollama, vLLM, llama.cpp) do not support
    SAE integration

-   Ollama requires specially packaged models rather than raw Hugging
    Face weights

-   Behavioral modification requires extensive system prompts that
    consume context window space

-   Fine-tuning for behavioral changes is resource-intensive and
    inflexible

-   There is no practical way to experiment with feature steering in a
    production-like chat environment

**2.2 Desired Future State**

miLLM will provide a server that:

-   Loads raw Hugging Face models without preprocessing or conversion

-   Integrates SAEs at configurable model layers

-   Allows real-time adjustment of feature activation strengths

-   Exposes an OpenAI-compatible API for seamless integration with
    existing tools (e.g., Open WebUI)

-   Provides monitoring capabilities for observing feature activations
    during inference

**3. Business Objectives**

  -------- ----------------------------------- ---------------------------
  **ID**   **Objective**                       **Success Metric**

  BO-1     Enable practical application of SAE Users can successfully
           steering in local inference         steer model outputs using
           environments                        SAE features

  BO-2     Reduce dependency on system prompts Equivalent behavioral
           for behavioral control              modifications achieved with
                                               \<10% context usage

  BO-3     Provide seamless integration with   100% compatibility with
           existing LLM tooling ecosystem      OpenAI API clients

  BO-4     Support research into real-world    System can demonstrate both
           implications of feature steering    monitoring and influence
                                               scenarios

  BO-5     Create foundation for miStudio      Defined API contract for
           integration pipeline                configuration exchange with
                                               miStudio
  -------- ----------------------------------- ---------------------------

**4. Stakeholders and Users**

**4.1 Primary User Persona: Developer/Researcher**

Technical users who want to integrate SAE-steered models into their
applications or research workflows. They are comfortable with APIs,
Docker, and Python environments. They seek fine-grained control over
model behavior and the ability to experiment with interpretability
techniques in practical settings.

**4.2 Secondary User Personas**

**4.2.1 MI Researchers**

Academics and researchers exploring mechanistic interpretability in
laboratory or experimental settings. They require detailed activation
monitoring and the ability to test hypotheses about feature effects.

**4.2.2 Power Users/Hobbyists**

Enthusiasts running local LLMs who want advanced control over model
behavior beyond what standard tools provide. They value ease of setup
and integration with existing chat interfaces.

**5. Scope**

**5.1 In Scope (Version 1.0)**

-   OpenAI API-compatible endpoints: /v1/chat/completions,
    /v1/completions, /v1/models, /v1/embeddings

-   Streaming response support for chat applications

-   Hugging Face model downloading and loading (Transformers format)

-   SAE downloading from Hugging Face

-   Single SAE attachment to configurable model layer

-   Multiple feature adjustment within attached SAE

-   Output steering (modify generation based on feature strengths)

-   Input monitoring (observe feature activations without modifying
    output)

-   Input monitoring on embeddings endpoint

-   Administrative web UI for model/SAE management and feature
    configuration

-   Real-time feature activation monitoring in UI

-   Configurable feature monitoring selection

-   Saveable steering configuration profiles (UI and API selectable)

-   Docker containerization

-   4-bit and 8-bit quantization support via bitsandbytes

-   Request queuing for single-user, multiple-request scenarios

**5.2 Out of Scope (Version 1.0)**

-   Multi-user authentication and access control

-   Multiple concurrent SAEs across different layers

-   GGUF or other non-Transformers model formats

-   Kubernetes deployment configurations

-   Feature discovery or analysis tools (delegated to miStudio)

-   Neuronpedia API integration

-   Direct miStudio push integration

**5.3 Future Considerations**

-   Multi-layer SAE support with coordinated feature adjustment

-   Additional model format support (GGUF, etc.)

-   API key authentication for non-local deployments

-   miStudio direct push integration

-   Neuronpedia integration for feature browsing

-   Multi-user request management

**6. Functional Requirements**

**6.1 Model Management**

  -------- ----------------------------------------- ---------------------
  **ID**   **Requirement**                           **Priority**

  FR-1.1   System shall download models from Hugging Must Have
           Face by model identifier                  

  FR-1.2   System shall load models in Hugging Face  Must Have
           Transformers format (safetensors/pytorch) 

  FR-1.3   System shall support 4-bit and 8-bit      Must Have
           quantization via bitsandbytes             

  FR-1.4   System shall cache downloaded models      Must Have
           locally                                   

  FR-1.5   System shall display model memory         Should Have
           requirements before loading               

  FR-1.6   System shall support any model format     Should Have
           loadable by Transformers library          
           (extensibility)                           
  -------- ----------------------------------------- ---------------------

**6.2 SAE Management**

  -------- ----------------------------------------- ---------------------
  **ID**   **Requirement**                           **Priority**

  FR-2.1   System shall download SAEs from Hugging   Must Have
           Face by identifier                        

  FR-2.2   System shall attach a single SAE to a     Must Have
           specified model layer                     

  FR-2.3   System shall allow                        Must Have
           detachment/reattachment of SAEs without   
           server restart                            

  FR-2.4   System shall cache downloaded SAEs        Must Have
           locally                                   

  FR-2.5   Architecture shall support future         Must Have
           multi-SAE, multi-layer configurations     
  -------- ----------------------------------------- ---------------------

**6.3 Feature Steering**

  -------- ----------------------------------------- ---------------------
  **ID**   **Requirement**                           **Priority**

  FR-3.1   System shall allow adjustment of          Must Have
           individual feature activation strengths   
           by feature index                          

  FR-3.2   System shall support simultaneous         Must Have
           adjustment of multiple features           

  FR-3.3   System shall apply steering to model      Must Have
           output generation                         

  FR-3.4   System shall allow feature adjustments    Must Have
           without server restart                    

  FR-3.5   System shall support both positive        Must Have
           (amplify) and negative (suppress)         
           steering values                           
  -------- ----------------------------------------- ---------------------

**6.4 Input Monitoring**

  -------- ----------------------------------------- ---------------------
  **ID**   **Requirement**                           **Priority**

  FR-4.1   System shall capture feature activations  Must Have
           for incoming requests                     

  FR-4.2   System shall expose activation data via   Must Have
           monitoring API/websocket                  

  FR-4.3   System shall support monitoring on        Must Have
           embeddings endpoint without modifying     
           output                                    

  FR-4.4   System shall allow user-configurable      Must Have
           selection of which features to monitor    
  -------- ----------------------------------------- ---------------------

**6.5 API Compatibility**

  -------- ----------------------------------------- ---------------------
  **ID**   **Requirement**                           **Priority**

  FR-5.1   System shall implement                    Must Have
           /v1/chat/completions endpoint per OpenAI  
           specification                             

  FR-5.2   System shall implement /v1/completions    Must Have
           endpoint per OpenAI specification         

  FR-5.3   System shall implement /v1/models         Must Have
           endpoint to list available models         

  FR-5.4   System shall implement /v1/embeddings     Must Have
           endpoint                                  

  FR-5.5   System shall support streaming responses  Must Have
           (SSE) for chat/completions                

  FR-5.6   System shall be compatible with any       Must Have
           OpenAI API-compatible client (e.g., Open  
           WebUI, LibreChat, etc.)                   
  -------- ----------------------------------------- ---------------------

**6.6 Configuration Management**

  -------- ----------------------------------------- ---------------------
  **ID**   **Requirement**                           **Priority**

  FR-6.1   System shall persist steering             Must Have
           configurations as named profiles          

  FR-6.2   System shall allow profile selection via  Must Have
           admin UI                                  

  FR-6.3   System shall allow profile selection via  Must Have
           API parameter                             

  FR-6.4   System shall support import/export of     Must Have
           profiles for miStudio compatibility       

  FR-6.5   Profile format shall follow documented    Must Have
           contract for miStudio interoperability    
  -------- ----------------------------------------- ---------------------

**6.7 Administrative UI**

  -------- ----------------------------------------- ---------------------
  **ID**   **Requirement**                           **Priority**

  FR-7.1   UI shall provide interface for model      Must Have
           download and selection                    

  FR-7.2   UI shall provide interface for SAE        Must Have
           download and attachment                   

  FR-7.3   UI shall provide interface for feature    Must Have
           value adjustment                          

  FR-7.4   UI shall display real-time feature        Must Have
           activation values during inference        

  FR-7.5   UI shall allow configuration of which     Must Have
           features to monitor                       

  FR-7.6   UI shall provide profile management       Must Have
           (create, edit, delete, activate)          

  FR-7.7   UI shall display server status and loaded Must Have
           model information                         
  -------- ----------------------------------------- ---------------------

**7. Non-Functional Requirements**

**7.1 Performance**

  --------- ----------------------------------------- ---------------------
  **ID**    **Requirement**                           **Target**

  NFR-1.1   SAE hook overhead shall not significantly \<15% overhead vs
            degrade inference latency                 base model

  NFR-1.2   System shall handle queued requests       Graceful queuing for
            without dropping connections              5+ pending requests

  NFR-1.3   Streaming response latency (time to first \<500ms after model
            token)                                    loaded
  --------- ----------------------------------------- ---------------------

**7.2 Reliability**

  --------- ----------------------------------------- ---------------------
  **ID**    **Requirement**                           **Behavior**

  NFR-2.1   Configuration errors (bad SAE, invalid    Immediate error, no
            feature index) shall fail fast with clear partial state
            messages                                  

  NFR-2.2   Runtime errors (OOM) shall degrade        Disable SAE, continue
            gracefully when possible                  base model

  NFR-2.3   System shall log all errors with          Structured logging
            sufficient context for debugging          with stack traces
  --------- ----------------------------------------- ---------------------

**7.3 Deployability**

  --------- ----------------------------------------- ---------------------
  **ID**    **Requirement**                           **Target**

  NFR-3.1   System shall be deployable via Docker     Single docker-compose
            container                                 up

  NFR-3.2   System shall be runnable without Docker   pip install + python
            for development                           run

  NFR-3.3   Docker image shall support NVIDIA GPU     CUDA-enabled base
            passthrough                               image

  NFR-3.4   Configuration shall be manageable via     12-factor app
            environment variables                     compliance
  --------- ----------------------------------------- ---------------------

**7.4 Security**

  --------- ----------------------------------------- ---------------------
  **ID**    **Requirement**                           **Note**

  NFR-4.1   System assumes trusted local network (no  Similar to Ollama
            authentication in v1)                     model

  NFR-4.2   Architecture shall support future         Design for
            addition of API key authentication        extensibility

  NFR-4.3   UI shall not expose system paths or       Abstracted model/SAE
            sensitive configuration details           identifiers
  --------- ----------------------------------------- ---------------------

**8. Integration Requirements**

**8.1 Hugging Face**

-   Download models via huggingface_hub library

-   Support private model access via HF_TOKEN environment variable

-   Cache models in configurable local directory

**8.2 OpenAI API Client Compatibility**

-   System shall be configurable as a backend for any OpenAI
    API-compatible client

-   All standard chat functionality shall work without client-side
    modification

-   Tested examples should include Open WebUI, LibreChat, and similar
    tools

**8.3 miStudio Integration**

miLLM and miStudio shall communicate via a defined file/API contract:

-   Profile export format: JSON schema defining model, SAE, layer, and
    feature configurations

-   Profile import: miLLM shall validate and load miStudio-exported
    configurations

-   Future: Direct push API for miStudio to update running miLLM
    instance

**9. Technical Constraints**

-   Backend: Python (FastAPI) - required for PyTorch/Transformers
    ecosystem compatibility

-   Frontend: Modern web framework (React, Vue, or Svelte) - TBD in PRD

-   Model Loading: Hugging Face Transformers library

-   SAE Framework: Compatible with TransformerLens or equivalent hooking
    mechanism

-   Quantization: bitsandbytes library for 4-bit/8-bit support

-   Container Runtime: Docker with NVIDIA Container Toolkit support

-   GPU: NVIDIA CUDA-compatible GPU required for reasonable performance

**10. Risks and Mitigations**

  ----------------------- ---------------- -----------------------------------
  **Risk**                **Likelihood**   **Mitigation**

  SAE hooking introduces  Medium           Benchmark early; provide option to
  unacceptable latency                     disable SAE for baseline comparison

  SAE format              High             Initially target Gemma-Scope SAEs
  incompatibilities                        only; document supported formats
  across sources                           

  Memory exhaustion with  Medium           Require quantization for large
  large models + SAE                       models; display memory estimates

  OpenAI API spec         Low              Target stable v1 endpoints;
  drift/incompatibility                    integration test against Open WebUI

  Misuse of steering for  Medium           Document ethical considerations;
  harmful manipulation                     this tool also demonstrates the
                                           risk
  ----------------------- ---------------- -----------------------------------

**11. Success Criteria**

Version 1.0 will be considered successful when:

-   A user can download a Gemma 2 model from Hugging Face via the UI

-   A user can download and attach a Gemma-Scope SAE to the loaded model

-   A user can adjust feature values (e.g., the \"yelling\" feature) and
    observe changed outputs

-   An OpenAI API-compatible client (e.g., Open WebUI) can successfully
    connect to miLLM and conduct a chat conversation

-   Feature activations are visible in real-time in the admin UI during
    inference

-   Steering configurations can be saved, loaded, and selected via API

-   The system runs successfully in a Docker container with GPU
    passthrough

**12. Glossary**

  ----------------- -----------------------------------------------------
  **Term**          **Definition**

  SAE               Sparse Autoencoder - a neural network trained to
                    decompose model activations into interpretable
                    features

  Feature           A learned direction in activation space that
                    corresponds to a human-interpretable concept

  Steering          Modifying model behavior by adjusting feature
                    activation strengths during inference

  Gemma-Scope       A project that trained SAEs on Gemma 2 models and
                    published them with feature annotations

  Neuronpedia       A platform hosting visualizations and labels for SAE
                    features

  Hooking           Intercepting model activations at a specific layer to
                    read or modify them

  miStudio          Companion application for SAE training, feature
                    discovery, and multi-feature steering experiments
  ----------------- -----------------------------------------------------

**Appendix A: Example Use Case**

**Scenario: Demonstrating Feature Steering**

A researcher wants to demonstrate how SAE steering can influence model
outputs:

-   1\. Launch miLLM and access the admin UI

-   2\. Download google/gemma-2-2b from Hugging Face

-   3\. Download the corresponding Gemma-Scope SAE for layer 12

-   4\. Attach the SAE to the loaded model

-   5\. Locate feature #1234 (labeled \"yelling/capitalization\" in
    Neuronpedia)

-   6\. Set feature #1234 strength to +5.0

-   7\. Save this configuration as profile \"yelling-demo\"

-   8\. Configure an OpenAI API-compatible client (e.g., Open WebUI) to
    use miLLM as its backend

-   9\. Send a chat message; observe that responses come back in ALL
    CAPS

-   10\. Return to admin UI; observe feature #1234 activation values
    during the conversation
