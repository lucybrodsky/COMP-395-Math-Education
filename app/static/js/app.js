/* ── State ──────────────────────────────────────────────────────── */
let mode = "practice";
let messages = [];       // [{role: "user"|"assistant", content: "..."}]
let currentEquation = null;  // active practice problem equation
let isStreaming = false;
let katexReady = false;  // set to true by KaTeX onload in HTML

/* ── DOM refs ───────────────────────────────────────────────────── */
const chatArea       = document.getElementById("chat-area");
const userInput      = document.getElementById("user-input");
const btnSend        = document.getElementById("btn-send");
const btnPractice    = document.getElementById("btn-practice");
const btnChat        = document.getElementById("btn-chat");
const practicePanel  = document.getElementById("practice-panel");
const problemText    = document.getElementById("problem-text");
const diffSelect     = document.getElementById("difficulty-select");
const topicSelect    = document.getElementById("topic-select");
const btnNewProblem  = document.getElementById("btn-new-problem");
const toolIndicator  = document.getElementById("tool-indicator");
const toolLabel      = document.getElementById("tool-label");

/* ── Tool display names ─────────────────────────────────────────── */
const TOOL_LABELS = {
    solve_linear_equation:   "Solving equation…",
    solve_quadratic_equation: "Solving quadratic…",
    solve_system_of_equations: "Solving system…",
    check_student_step:      "Checking your step…",
    generate_practice_problem: "Generating problem…",
    simplify_expression:     "Simplifying expression…",
    graph_function:          "Generating graph…",
    graph_equation:          "Generating graph…",
};

/* ── KaTeX render ───────────────────────────────────────────────── */
function renderMath(el) {
    if (typeof renderMathInElement !== "undefined") {
        renderMathInElement(el, {
            delimiters: [
                { left: "$$", right: "$$", display: true  },
                { left: "$",  right: "$",  display: false },
            ],
            throwOnError: false,
        });
    }
}

/* ── Escape HTML (preserve $ for KaTeX) ────────────────────────── */
function escHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

/* Apply light markdown: bold, newlines → <br> */
function formatContent(text) {
    let s = escHtml(text);
    s = s.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\n/g, "<br>");
    return s;
}

/* ── Mode switching ─────────────────────────────────────────────── */
function setMode(newMode) {
    mode = newMode;
    messages = [];
    currentEquation = null;
    clearChat();

    btnPractice.classList.toggle("active", newMode === "practice");
    btnChat.classList.toggle("active",     newMode === "chat");
    practicePanel.classList.toggle("hidden", newMode !== "practice");

    if (newMode === "practice") {
        userInput.placeholder = "Type your equation or step…";
        problemText.textContent = "—";
        addTutorMessage(
            "Welcome to **Practice Mode**!\n" +
            "Choose a topic and difficulty, then click **New Problem** to get a problem to solve."
        );
    } else {
        userInput.placeholder = "Ask me any math question…";
        addTutorMessage(
            "Hi! I'm your math tutor. I can help with linear equations, quadratic equations, " +
            "systems of equations, polynomials, and more.\n\n" +
            "Try asking: *How do I solve $x^2 - 5x + 6 = 0$?*\n" +
            "Or say: **graph y = x^2 - 4** to see a plot."
        );
    }
}

/* ── Clear chat area ────────────────────────────────────────────── */
function clearChat() {
    const welcome = document.getElementById("welcome-msg");
    chatArea.innerHTML = "";
    if (welcome) chatArea.appendChild(welcome);
    if (welcome) welcome.style.display = "none";
}

/* ── Add a completed tutor message ─────────────────────────────── */
function addTutorMessage(content) {
    const { bubble } = createMessageEl("tutor");
    bubble.innerHTML = formatContent(content);
    renderMath(bubble);
    scrollToBottom();
    return bubble;
}

/* ── Create message DOM structure ───────────────────────────────── */
function createMessageEl(role) {
    const msgEl = document.createElement("div");
    msgEl.className = `message ${role}`;

    const roleEl = document.createElement("div");
    roleEl.className = "message-role";
    roleEl.textContent = role === "user" ? "You" : "Tutor";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    msgEl.appendChild(roleEl);
    msgEl.appendChild(bubble);
    chatArea.appendChild(msgEl);
    return { msgEl, bubble };
}

/* ── Scroll to latest message ───────────────────────────────────── */
function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
}

/* ── New Problem ────────────────────────────────────────────────── */
async function newProblem() {
    if (isStreaming) return;
    btnNewProblem.disabled = true;

    try {
        const res = await fetch("/new-problem", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                difficulty: diffSelect.value,
                topic:      topicSelect ? topicSelect.value : "linear",
            }),
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const problem = await res.json();

        currentEquation = problem.equation;
        problemText.textContent = problem.equation;

        // Reset conversation with context about the new problem
        messages = [];
        clearChat();

        const intro =
            `Let's solve this problem together:\n\n` +
            `$$${problem.equation}$$\n\n` +
            `Take a look and tell me: what's your first step?`;

        addTutorMessage(intro);

        // Seed history so the LLM has the problem context from the start
        messages = [
            { role: "user",      content: `I want to practice solving: ${problem.equation}` },
            { role: "assistant", content: intro },
        ];
    } catch (err) {
        addTutorMessage("Sorry, I couldn't generate a problem. Is the server running?");
        console.error(err);
    } finally {
        btnNewProblem.disabled = false;
        userInput.focus();
    }
}

/* ── Send a message ─────────────────────────────────────────────── */
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isStreaming) return;

    // In practice mode, nudge the user to get a problem first
    if (mode === "practice" && messages.length === 0) {
        addTutorMessage("Click **New Problem** to get a problem to work on first!");
        return;
    }

    userInput.value = "";
    setInputEnabled(false);

    // Show user bubble
    const { bubble: userBubble } = createMessageEl("user");
    userBubble.textContent = text;
    scrollToBottom();
    messages.push({ role: "user", content: text });

    // Prepare streaming tutor bubble with immediate "Thinking…" placeholder
    const { bubble: tutorBubble } = createMessageEl("tutor");
    tutorBubble.classList.add("streaming");
    tutorBubble.dataset.raw = "";
    tutorBubble.textContent = "Thinking…";
    scrollToBottom();

    let fullContent = "";
    let graphData = null;   // set when event: graph arrives

    try {
        const res = await fetch("/chat", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ messages, mode, equation: currentEquation }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let   buffer  = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // SSE events are separated by \n\n
            const parts = buffer.split("\n\n");
            buffer = parts.pop(); // keep trailing incomplete event

            for (const block of parts) {
                if (!block.trim()) continue;
                const { eventType, data } = parseSSEBlock(block);

                // Handle graph event inline (before delegating to handleEvent)
                if (eventType === "graph") {
                    if (data && data.image_b64) {
                        graphData = data;
                        // Show the graph immediately while text tokens stream in below
                        tutorBubble.textContent = "";
                        tutorBubble.dataset.raw = "";
                        const img = document.createElement("img");
                        img.src = `data:image/png;base64,${data.image_b64}`;
                        img.alt = data.expression ? `Graph of ${data.expression}` : "Graph";
                        img.className = "graph-img";
                        tutorBubble.appendChild(img);
                        tutorBubble.appendChild(document.createElement("br"));
                        scrollToBottom();
                    }
                    continue;
                }

                fullContent = handleEvent(eventType, data, tutorBubble, fullContent);
            }
        }

    } catch (err) {
        tutorBubble.innerHTML = "<em>Something went wrong — please try again.</em>";
        console.error(err);
    } finally {
        tutorBubble.classList.remove("streaming");
        const raw = fullContent || tutorBubble.dataset.raw || "";

        // Compose final bubble: graph image (if any) + formatted text
        // Clear interim streaming content and rebuild cleanly
        tutorBubble.innerHTML = "";

        if (graphData && graphData.image_b64) {
            const img = document.createElement("img");
            img.src = `data:image/png;base64,${graphData.image_b64}`;
            img.alt = graphData.expression ? `Graph of ${graphData.expression}` : "Graph";
            img.className = "graph-img";
            tutorBubble.appendChild(img);
        }

        if (raw) {
            const textDiv = document.createElement("div");
            textDiv.innerHTML = formatContent(raw);
            tutorBubble.appendChild(textDiv);
            renderMath(tutorBubble);
        } else if (!graphData) {
            tutorBubble.innerHTML = "<em style='color:#dc2626'>No response — check that Ollama is running and the model is pulled.</em>";
        }

        toolIndicator.classList.add("hidden");
        scrollToBottom();

        if (fullContent) {
            messages.push({ role: "assistant", content: fullContent });
        }

        setInputEnabled(true);
        userInput.focus();
    }
}

/* ── Parse a single SSE event block ─────────────────────────────── */
function parseSSEBlock(block) {
    let eventType = "message";
    let rawData   = "";

    for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
            rawData = line.slice(6);
        }
    }

    let data = rawData;
    try { data = JSON.parse(rawData); } catch (_) { /* keep as string */ }

    return { eventType, data };
}

/* ── Handle a parsed SSE event ──────────────────────────────────── */
function handleEvent(eventType, data, bubble, fullContent) {
    switch (eventType) {
        case "token": {
            const chunk = typeof data === "string" ? data : String(data);
            // Clear the "Thinking…" placeholder on first real token
            if (!bubble.dataset.raw) bubble.textContent = "";
            bubble.dataset.raw = (bubble.dataset.raw || "") + chunk;
            // Show plain text while streaming; format on "done"
            bubble.textContent = bubble.dataset.raw;
            fullContent += chunk;
            scrollToBottom();
            break;
        }
        case "tool": {
            const name  = data && data.name ? data.name : "";
            const label = TOOL_LABELS[name] || "Thinking…";
            toolLabel.textContent = label;
            toolIndicator.classList.remove("hidden");
            break;
        }
        case "done": {
            toolIndicator.classList.add("hidden");
            break;
        }
        case "error": {
            const msg = typeof data === "string" ? data : "An error occurred.";
            bubble.innerHTML = `<em style="color:#dc2626">${escHtml(msg)}</em>`;
            bubble.classList.remove("streaming");
            toolIndicator.classList.add("hidden");
            break;
        }
    }
    return fullContent;
}

/* ── Toggle input controls ──────────────────────────────────────── */
function setInputEnabled(enabled) {
    isStreaming      = !enabled;
    userInput.disabled = !enabled;
    btnSend.disabled   = !enabled;
}

/* ── Keyboard submit ────────────────────────────────────────────── */
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

/* ── Initialise ─────────────────────────────────────────────────── */
(function init() {
    practicePanel.classList.remove("hidden");

    const welcome = document.getElementById("welcome-msg");
    if (welcome) welcome.style.display = "";

    userInput.placeholder = "Type your answer or question…";
})();
