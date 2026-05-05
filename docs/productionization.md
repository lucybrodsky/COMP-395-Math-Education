# Productionizing the Algebra Tutor — A Deployment Pitch

This document outlines how to take the COMP-395 math tutor from a locally running prototype to a
production-grade application suitable for a school district or ed-tech company.

---

## Hosting Architecture

The app today runs as a Flask server on a developer's laptop, with Ollama serving the LLM locally.
Neither of those is deployable at scale.

**Why not serverless (AWS Lambda / API Gateway)?**
The tutor uses Server-Sent Events (SSE) to stream tokens to the student in real time. SSE requires a
persistent HTTP connection that can remain open for tens of seconds. AWS Lambda's maximum response
timeout is 29 seconds, and API Gateway imposes its own limits — both kill a mid-stream tutoring
response. Serverless is the wrong shape for this workload.

**Recommended stack: containerized Flask on Fly.io**
Fly.io deploys Docker containers globally, supports long-lived HTTP connections natively, and can
scale to zero during off-peak hours. This matters because usage is highly bursty — concentrated
during school hours (roughly 8 am–3 pm on weekdays) with near-zero traffic overnight and on
weekends. Scale-to-zero means we pay only for active sessions, not idle uptime.

The deployment topology:

- **Fly.io** — Flask container (2 shared vCPU, 512 MB RAM per instance; autoscale 0–N)
- **Upstash Redis** — lightweight session store for conversation history (Fly add-on, ~$0/mo at low scale)
- **Cloudflare CDN** — serve static JS/CSS assets at the edge

For institutions already in the AWS ecosystem, an equivalent stack is **ECS Fargate** behind an
**Application Load Balancer**, which pairs naturally with AWS Bedrock for model inference (see below).
Fargate removes the need to manage EC2 instances while preserving long-lived connection support.

---

## Model Choice at Scale

The prototype uses gemma3:12b via Ollama — a self-hosted model that cannot be deployed to a shared
cloud environment without provisioning dedicated GPU infrastructure.

**The key insight**: SymPy already handles all math computation. The LLM only needs to be a good
*tutor* — explaining concepts, asking Socratic questions, and adapting to student responses. That
does not require a frontier model.

**Recommended options, in order of preference:**

| Model | Route | Strength |
|---|---|---|
| GPT-4o-mini | OpenAI API | Lowest latency, cheapest |
| Llama 3 8B | AWS Bedrock | AWS-native, data never leaves your account |
| Claude Haiku 4.5 | AWS Bedrock or Anthropic API | Best instruction-following for pedagogy |

**AWS Bedrock** is the strongest choice for institutional deployments. Bedrock is HIPAA-eligible,
SOC 2 Type II certified, and — critically — student conversation data is **not used to train
foundation models**. This satisfies the data processing requirements most school districts will
demand before signing a vendor agreement. Bedrock also consolidates billing into existing AWS EDU
spend, simplifying procurement.

For K-12 deployments specifically, Bedrock Llama 3 8B keeps the entire inference pipeline inside the
institution's AWS account, which avoids the third-party data sharing triggers in COPPA and SOPIPA
(see Data Privacy below).

---

## Data Privacy

The current prototype stores nothing — all conversation state lives in the browser's JavaScript
memory and is discarded on page refresh. That is a good starting point, but production requires
deliberate choices.

**In production, conversations must be:**
- Encrypted at rest (AES-256) and in transit (TLS 1.3)
- Accessible only to the student who created them, their instructor, and platform admins — enforced
  via role-based access control
- Retained for a configurable window (default: 90 days) then automatically purged
- Deletable on student request (right to erasure)

**K-12 context (stricter):**
- **COPPA** (Children's Online Privacy Protection Act): for students under 13, verifiable parental
  consent is required before collecting any personal data. Minimize collection to what is strictly
  necessary.
- **SOPIPA** (Student Online Personal Information Protection Act, and analogous state laws): prohibits
  selling student data, using it for targeted advertising, or sharing it with third parties for
  non-educational purposes.
- Practical implication: the LLM API call itself is a data transfer to a third party. For K-12,
  AWS Bedrock (with a signed Business Associate Agreement) or a fully self-hosted model is the only
  compliant path.

**Higher education context (somewhat more flexible):**
- **FERPA** governs educational records — students have the right to inspect their own records, and
  institutions control who else can access them.
- A standard data processing addendum with the LLM provider typically satisfies FERPA, making direct
  API calls to OpenAI or Anthropic viable as long as PII is minimized in prompts.

---

## Cost Model

Assumptions: 1 tutoring session per user per day, 20 message exchanges per session, ~1,500 input
tokens and ~250 output tokens per exchange (input grows as conversation history accumulates).

| Scale | Input/mo | Output/mo | GPT-4o-mini | Bedrock Llama 3 8B | Bedrock Claude Haiku |
|---|---|---|---|---|---|
| 100 DAU | 90M tokens | 15M tokens | ~$22/mo | ~$23/mo | ~$41/mo |
| 1,000 DAU | 900M tokens | 150M tokens | ~$225/mo | ~$230/mo | ~$413/mo |

Hosting (Fly.io or ECS Fargate) adds roughly $20–80/mo at 100 DAU and $150–400/mo at 1,000 DAU
depending on concurrency peaks.

**Where does it become unsustainable?**
At 100–1,000 DAU, all three model options are easily covered by $5/student/month institutional
licensing — the LLM cost is a fraction of revenue. The ceiling emerges when you move to frontier
models (Claude Sonnet, GPT-4o full) at 5,000+ DAU without negotiated volume pricing; costs reach
$10,000+/month before operational overhead is factored in. The mitigation is to stay on efficient
small models (Llama 3 8B, GPT-4o-mini) and negotiate AWS EDU credits or OpenAI nonprofit pricing
before scaling past 2,000 DAU.

---

## Failure Modes

**1. LLM API is down**
Apply a circuit breaker: after two consecutive API failures, stop sending requests for 60 seconds
and show the student a clear message ("The tutor is temporarily unavailable — try again in a moment").
A small cache of pre-generated problem hints can fill the gap for common problem types. Retry with
exponential backoff once the circuit resets.

**2. The model gives a mathematically wrong answer**
This is the highest-stakes failure in an educational product. The app already has a strong
mitigation: **SymPy validates every student step**. The `check_student_step` and
`check_system_answer` functions in `math_tools.py` verify algebraic correctness independently of
the LLM — the model cannot hallucinate its way past a SymPy check. In production, extend this by
flagging any LLM response that contradicts the SymPy verdict and routing that session to a human
review queue. Track the flag rate per model version as a regression metric.

**3. A student submits harmful or off-topic input**
The system prompt already includes jailbreak protection. In production, add a **moderation layer
before the LLM call**: run the student's message through AWS Bedrock's Llama Guard (if using
Bedrock) or OpenAI's Moderation API. Messages that exceed the harm threshold are blocked and logged
without reaching the model. Rate-limit aggressively (e.g., 60 messages per hour per user) to
prevent brute-force prompt injection. For repeated violations, escalate to a human moderator and
optionally notify the student's instructor.
