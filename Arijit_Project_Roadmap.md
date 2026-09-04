# Project Roadmap for Arijit Bhattacharyya
### AI + Full-Stack portfolio built for internships & placements

*Prepared August 2026 · Direction: AI + Full-Stack blend · Target: keep options open (product / startup / service) · Scope: 1 flagship + 2 medium + 2 small*

---

## The one idea that matters most

Recruiters and interviewers almost never reject you because your project *idea* wasn't clever. They lose interest because the project is **not shipped, not original, and shows no engineering judgment**. A "Spotify clone" and a "bank management system" are low-weight not because they're bad, but because 50,000 other students submit the exact same thing, un-deployed, with a one-line README.

So the goal of this roadmap is not just *what to build* — it's *how to build it so it carries weight*. A well-executed medium project beats a sloppy flagship every time.

### The 6 signals that turn any project into a "high-weightage" project

1. **It's live.** A working URL beats a GitHub repo every time. "Here, try it" is the single strongest signal you can send.
2. **It has real data or real users.** Even 15 real users, or a real public dataset, beats fake seed data. It proves the thing actually works end-to-end.
3. **The README sells it.** A short problem statement, an architecture diagram, a demo GIF, screenshots, and clear setup steps. Most students skip this — which is exactly why doing it makes you stand out.
4. **It shows engineering hygiene.** Tests, a CI pipeline (GitHub Actions), meaningful commit history over weeks (not one giant "final commit"), environment configs. This signals "will write maintainable code," which is what a job actually is.
5. **It has numbers.** "96% accuracy," "handles 500 req/s," "cut load time from 4s to 900ms." Metrics make a bullet impossible to skim past.
6. **You've written about it.** A short blog post or LinkedIn write-up explaining *why* you made a design decision proves you understand your own project. Interviewers can smell a tutorial-follower vs. a builder.

If you do nothing else, apply these six to **every** project below.

---

## Honest audit of your current resume

| Project | Verdict | Action |
|---|---|---|
| Java Bank Management System | Very low weight (first-year staple, console-only) | Drop, or merge into a single "Core Java / OOP fundamentals" line |
| ATM Simulation in Java | Very low weight (near-duplicate of the above) | Drop, or merge as above |
| Crescent — Online Grocery (JS/Mongo/Express) | **Genuine full-stack — your best current project.** Under-sold. | Keep & upgrade: deploy it, add auth + payments (Stripe test mode), write real bullets with metrics |
| Spotify Clone (React/Tailwind) | Low-to-medium (clone = follows a pattern, no original problem) | Keep only as a supporting "frontend skills" line, or fold into your portfolio site |
| **AI Corn/Maize Disease Detection (research)** | **Your single biggest differentiator.** Very few 2nd-years have real ML + XAI research. | **Extend it into the flagship below** — this is the move |

The theme: you already have the raw material for a standout portfolio. We're going to consolidate the weak projects, and turn your research into something recruiters can actually click on.

---

## THE ROADMAP

### Flagship — "CropGuard AI": an explainable crop-disease diagnosis platform

**Why this is the right flagship for you:** it converts your ongoing research (which currently lives as one resume line and a repo of notebooks) into a **deployed, explainable, full-stack product**. It simultaneously shows ML, XAI, full-stack, cloud deployment, *and* product thinking — and it comes with a story no clone can match: *"I took my research and shipped it as something a farmer could actually use."* That single sentence will carry an entire interview.

**What it does:** a user (farmer / agronomist / student) uploads a photo of a plant leaf → a CNN classifies the disease with a confidence score → a **Grad-CAM heatmap overlays the exact diseased region** (this is your XAI angle — you show *why* the model decided, not just *what*) → the app returns plain-language treatment recommendations → results are saved to the user's history.

**Recommended architecture (this specific split is what impresses):**
- **Frontend:** React + Tailwind (your stack), mobile-first / PWA — because the real users are on phones in fields. Add `i18n` for a local language; accessibility like this reads as genuine product sense.
- **API / app server:** Node.js + Express (your stack) — handles auth, users, history, and talks to the ML service.
- **ML microservice:** Python + FastAPI serving the model. **Splitting inference into its own service is a real system-design decision** you can defend in an interview ("I isolated the GPU/CPU-heavy inference so the API stays responsive and each part scales independently").
- **Model:** transfer learning with MobileNetV2 / EfficientNet on the public **PlantVillage dataset** (~54k images, 38 classes) — 95%+ test accuracy is very achievable. XAI via **Grad-CAM / Grad-CAM++**.
- **Data:** MongoDB (users, prediction history) + object storage for images (Azure Blob Storage — leans on your **Azure Fundamentals cert**).
- **Deploy:** Dockerize everything; ship on **Azure** (Container Apps / App Service) to tie back to your cert, or Render/Railway + Hugging Face Spaces for the model if you want free hosting.

**Stretch goals (each one is a resume bullet + an interview talking point):**
- Result caching + a confidence threshold that returns "unsure — retake photo" instead of a bad guess (shows you think about failure modes).
- Rate limiting and async processing for large image uploads.
- Model versioning / a simple retraining pipeline (light MLOps).
- A small feedback loop where users confirm/correct predictions (data flywheel — great product story).

**Metrics to capture:** test accuracy, number of disease classes, inference latency, live URL, and (if you can get even 20 real testers) usage numbers.

*If you'd rather not tie the flagship to your research:* an equally strong alternative is an **AI-powered interview/DSA-prep platform** (upload a problem → AI hints without spoilers → tracks your weak patterns → spaced repetition), which shows the same full-stack + LLM + product skills. But I'd pick CropGuard — the research tie-in is a genuine edge.

---

### Medium 1 — A domain-specific RAG assistant (the most in-demand skill of 2026)

**Why:** Retrieval-Augmented Generation (RAG) — building apps on top of LLMs with your own data — is *the* skill companies are hiring for right now, and most students only ever call a chat API. Doing RAG *properly* (embeddings, chunking, a vector DB, citations, evaluation) puts you ahead of the pack.

**The trap to avoid:** "Chat with your PDF" is now a tutorial cliché. **Make it domain-specific** so it stands out — e.g., an assistant over your university's academic regulations + course catalog, or over a body of agriculture research (which pairs beautifully with your flagship), or over a specific area of law/medicine/finance.

**Tech:** React frontend, FastAPI or Express backend, an LLM (Gemini / OpenAI, or free via Groq / Ollama for local open models), embeddings (`sentence-transformers` or OpenAI embeddings), a vector DB (Chroma / pgvector / Pinecone), streaming responses.

**What makes it high-weight (not just another chatbot):** show **source citations** (highlight which chunk the answer came from), handle "I don't know" instead of hallucinating, and add a small **evaluation** (even a hand-labeled test set with accuracy) — evaluation is what separates someone who *understands* RAG from someone who copied a tutorial.

---

### Medium 2 — A scalable backend service (this is your system-design gym)

**Why:** you told me you're keeping product companies open, and this project is where you *build the intuition* that system-design rounds test. It's also the kind of project you can literally walk through in a design interview.

**Recommended build — a URL shortener with analytics.** It looks simple and is deceptively deep, letting you layer real concepts:
- Base62 short-code generation and collision handling
- **Redis caching** for hot links (and a cache-aside pattern you can explain)
- **Rate limiting** per API key
- Click **analytics** (geo, referrer, device) with a dashboard
- Clean REST API design + database indexing for fast lookups
- Load-testing it and reporting the numbers (e.g., "sustained ~800 req/s on a single instance")

**Alternative if you want something flashier:** a **real-time collaborative app** (shared notes / whiteboard / chat) using WebSockets (Socket.io) — this shows you understand real-time systems, presence, and optimistic UI. Pick the URL shortener for maximum system-design signal; pick the real-time app if you want a more visual demo.

---

### Small 1 — A real portfolio website

React + Tailwind (you already know these), deployed on Vercel/Netlify, one page, fast. Each project gets a short **case study** (problem → what you built → a design decision you're proud of → live link + repo), not just a screenshot. Recruiters *will* look for this, it takes a weekend, and it's where your blog write-ups live. High return for low effort.

### Small 2 — One "engineering hygiene" project (pick one)

- **A small, well-tested utility** — a browser extension or a CLI tool that does one useful thing — but with **unit tests, a GitHub Actions CI badge, and real docs**. The point isn't the tool; it's proving you can ship maintainable, tested code. ~95% of student repos have zero tests, so this is a cheap way to stand out.
- **OR your first open-source contributions** — find `good-first-issue` labels on active repos, get 2–3 PRs merged. A merged PR in a real project is a strong, verifiable signal and teaches you to read large codebases.

---

## Parallel tracks (not "projects," but they carry huge weight)

- **DSA — still your #1 interview priority.** Projects get you *shortlisted*; DSA gets you *through the interview*. Be consistent (a little daily beats cramming). Aim for solid pattern coverage (arrays, hashing, two pointers, sliding window, trees, graphs, DP, heaps) — quality over raw count, ~300–400 thoughtful problems over the next year.
- **CS fundamentals** — OS, DBMS, Computer Networks, OOP. Indian fresher interviews lean on these heavily; they're easy marks if you prepare them.
- **Low-Level Design (LLD)** — start this soon (more on why in the next section). Your OOP background makes it low-hanging fruit.

---

## How to write project bullets that actually land

Use the format: **Built/Engineered [what] using [tech], achieving [metric/impact].** Lead with a strong verb, name the tech, end with a number.

**Before (your current Crescent bullet):**
> Online grocery platform built using JavaScript, MongoDB, Express.js, and AJAX.

**After:**
> Built a full-stack grocery e-commerce app (React, Node/Express, MongoDB) with JWT auth, cart, and Stripe test-mode checkout; deployed on Render serving 20+ test users with sub-1s page loads.

**Flagship example bullets:**
> Engineered *CropGuard AI*, an explainable crop-disease diagnosis platform (React, FastAPI, MobileNetV2) that classifies 38 diseases at 96% test accuracy and overlays Grad-CAM heatmaps to visualize model reasoning.

> Architected a decoupled inference microservice and containerized the stack (Docker) on Azure, keeping API latency under 300ms while isolating GPU-bound model serving.

**RAG example bullet:**
> Built a domain-specific RAG assistant (React, FastAPI, Chroma, Gemini) with source-cited answers and a 30-question eval set, reaching 90% answer accuracy while cutting hallucinations via retrieval grounding.

Notice: every bullet has **tech + an action + a number**. Go back and rewrite *all* your bullets this way.

---

## "Is system design really asked more in interviews now?" — the honest answer

**Short answer: yes, but with an important asterisk for someone at your stage.** *(Based on my knowledge of hiring patterns through 2025; the direction of this trend has been consistent and is very unlikely to have reversed.)*

**What's actually true:**

- **For experienced engineers (SDE-2 and up), High-Level Design (HLD) is now a core, often make-or-break round** — "design Twitter," "design a rate limiter," load balancing, caching, sharding, CAP, queues. This has intensified over the last few years.
- **For freshers / new grads / interns, DSA is *still* the primary filter.** System design has *not* replaced it. Anyone telling you to skip DSA for system design is giving you bad advice for your stage.
- **But two things have genuinely risen for freshers, especially in India:**
  - **Low-Level Design (LLD) / OOP design** — "design a parking lot / elevator / BookMyShow / a splitwise" — designing clean, extensible *classes*. This is increasingly common and directly builds on your OOP strength.
  - **Machine-coding rounds** — build a small working system in 90–120 minutes, judged on clean, modular, extensible code. Flipkart, Swiggy, Uber, PhonePe, Razorpay and many startups run these.
- **Simplified HLD is creeping into new-grad loops** at top product companies as a *differentiator* round — not usually a hard filter yet, but a strong plus, and it's expected by the time you interview for full-time roles.

**Why the shift is happening:** grinding LeetCode is now table stakes — *everyone* does it, so it no longer separates candidates. On top of that, AI tools can solve standard DSA problems, so companies increasingly test **engineering judgment** (design, trade-offs, writing extensible code) because that's harder to fake and closer to the real job.

**Your concrete plan (2nd year → placements):**
1. **Keep DSA as priority #1.** It's still what gets you through interviews right now.
2. **Start LLD/OOP design in the next few months** — it's the highest-ROI new skill for you and your OOP background makes it easy. Practice the classic problems and one machine-coding build.
3. **Learn HLD fundamentals gradually over the next 12–18 months** — you don't need mastery now, just familiarity (caching, load balancing, SQL vs NoSQL, queues, rate limiting) before final-year placements.
4. **Let your projects do double duty** — this is exactly why Medium 2 (the scalable backend) is in your roadmap. Building a URL shortener with Redis caching and rate limiting *is* system-design practice you can talk about from real experience.

---

## Suggested sequence (you have good runway before full-time placements)

- **Months 1–3:** Flagship (CropGuard AI). Keep DSA daily. → immediately upgrades your resume's centerpiece.
- **Months 3–5:** Medium 1 (RAG assistant). Start LLD practice alongside.
- **Months 5–6:** Medium 2 (scalable backend). Build the portfolio site (Small 1).
- **Months 6–7:** Small 2 (tested utility or open source). Write blog posts for each project; rewrite all resume bullets.
- **Ongoing the whole time:** DSA, CS fundamentals, and (from ~month 3) LLD.

By the end you'll have: one deployed flagship with a research story, a RAG app showing the hottest 2026 skill, a system-design-flavored backend, a clean portfolio, and proof of engineering hygiene — plus DSA + design prep underway. That is a genuinely strong shortlisting profile across product, startup, and service companies.

---

## How I can help

I can work on any of these *with* you — for example: designing the CropGuard architecture in detail, writing the training + Grad-CAM code, scaffolding the React/Express/FastAPI apps, setting up Docker + CI, building the RAG pipeline, writing your READMEs and blog posts, or rewriting your resume bullets. Just tell me which one you want to start with and we'll go build it.
