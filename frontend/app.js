// Auto-detect API URL based on current host
// When served from the same origin, use relative paths
const API_URL = window.location.origin;

// Configuration
const CONFIG = {
    useSyncMode: true,  // Use synchronous endpoint (no Celery required)
    pollInterval: 5000,  // Poll every 5 seconds for async mode
    maxPollAttempts: 60  // Max polling attempts (5 minutes)
};

// DOM Elements
const form = document.getElementById("analyze-form");
const sections = {
    input: document.getElementById("input-section"),
    loading: document.getElementById("loading-section"),
    error: document.getElementById("error-section"),
    results: document.getElementById("results-section")
};

const loaders = {
    phase: document.getElementById("loading-phase"),
    subtext: document.getElementById("loading-subtext"),
    progress: document.getElementById("poll-progress")
};

let pollInterval;

// Tab Logic
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
        // Remove active class from all
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active", "slide-up"));
        
        // Add to clicked
        const tabId = e.target.getAttribute("data-tab");
        e.target.classList.add("active");
        
        const content = document.getElementById(`tab-${tabId}`);
        content.classList.remove("hidden");
        content.classList.add("active", "slide-up");
    });
});

// App flow logic
function showSection(sectionName) {
    Object.values(sections).forEach(s => s.classList.add("hidden"));
    sections[sectionName].classList.remove("hidden");
}

function resetUI() {
    if (pollInterval) clearInterval(pollInterval);
    showSection("input");
    loaders.progress.style.width = "0%";
}

// Form Submission
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const audio_url = document.getElementById("audio_url").value;
    const sop_template = document.getElementById("sop_template").value;
    const pii_redact = document.getElementById("pii_redact").checked;

    showSection("loading");
    loaders.phase.innerText = "Submitting to AI...";
    loaders.subtext.innerText = "Initializing connection with backend.";
    loaders.progress.style.width = "10%";

    try {
        // Choose endpoint based on sync mode
        const endpoint = CONFIG.useSyncMode ? `${API_URL}/analyze/sync` : `${API_URL}/analyze`;
        
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                audio_url,
                sop_template,
                enable_pii_redaction: pii_redact
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API Error: ${response.status}`);
        }

        const data = await response.json();
        
        if (CONFIG.useSyncMode) {
            // Sync mode: response contains the full result
            loaders.progress.style.width = "100%";
            loaders.phase.innerText = "Done!";
            
            if (data.status === "completed") {
                setTimeout(() => renderResults(data), 500);
            } else if (data.status === "failed") {
                showError(data.error || "Analysis failed.");
            } else {
                // Unexpected status, show as error
                showError(`Unexpected status: ${data.status}`);
            }
        } else {
            // Async mode: response contains task_id, need to poll
            const taskId = data.task_id;
            startPolling(taskId);
        }
        
    } catch (err) {
        console.error("Submit error:", err);
        showError(err.message || "Failed to submit audio. Check console for details.");
    }
});

function showError(msg) {
    showSection("error");
    document.getElementById("error-message").innerText = msg;
}

// Polling Logic
function startPolling(taskId) {
    let attempts = 0;
    
    pollInterval = setInterval(async () => {
        attempts++;
        loaders.phase.innerText = "Parsing Audio & Processing LLM...";
        
        // Fake progress bar logic for visual feedback
        const progress = Math.min(10 + (attempts * 2), 90);
        loaders.progress.style.width = `${progress}%`;
        loaders.subtext.innerText = `Analyzing segments (Attempt ${attempts})...`;

        try {
            const res = await fetch(`${API_URL}/result/${taskId}`);
            const data = await res.json();
            
            if (data.status === "completed") {
                clearInterval(pollInterval);
                loaders.progress.style.width = "100%";
                loaders.phase.innerText = "Done!";
                setTimeout(() => renderResults(data), 500);
            } else if (data.status === "failed") {
                clearInterval(pollInterval);
                showError(data.error || "Celery processing failed internally.");
            }
        } catch(err) {
            console.error("Poll error", err);
            // Don't fail immediately on a network blip during polling
            if(attempts > CONFIG.maxPollAttempts) {
                 clearInterval(pollInterval);
                 showError("Timed out waiting for backend.");
            }
        }
    }, CONFIG.pollInterval);
}

// Render Results
function renderResults(data) {
    showSection("results");
    
    // Dump Raw JSON
    document.getElementById("res-raw-json").textContent = JSON.stringify(data, null, 2);
    
    // Safety checks for objects
    const analytics = data.analytics || {};
    const sop = data.sop_validation || {};
    
    // 1. Stats Grid
    document.getElementById("res-compliance").innerText = sop.overall_compliance ? `${sop.overall_compliance.toFixed(1)}%` : "N/A";
    document.getElementById("res-sentiment").innerText = analytics.sentiment_timeline ? analytics.sentiment_timeline.overall_sentiment : (analytics.overall_sentiment || "Unknown");
    
    const diarization = (data.diarized_transcript && data.diarized_transcript.talk_ratio) 
        ? data.diarized_transcript : analytics.diarization;
        
    document.getElementById("res-talk-ratio").innerText = diarization && diarization.talk_ratio ? `${diarization.talk_ratio.toFixed(2)}` : "N/A";
    
    document.getElementById("res-duration").innerText = data.processing_time_seconds 
        ? `${data.processing_time_seconds.toFixed(1)}s` 
        : (analytics.call_duration_seconds ? `${analytics.call_duration_seconds}s` : "N/A");

    // 2. Summary & Findings
    document.getElementById("res-summary-text").innerText = data.summary || "No summary provided by LLM.";
    document.getElementById("res-summary-text").classList.remove("loading-pulse");
    
    const keywordsContainer = document.getElementById("res-keywords");
    keywordsContainer.innerHTML = "";
    if (data.keywords && Array.isArray(data.keywords)) {
        data.keywords.forEach(kw => {
            const span = document.createElement("span");
            span.className = "chip";
            span.innerText = kw;
            keywordsContainer.appendChild(span);
        });
    } else {
        keywordsContainer.innerText = "No keywords extracted.";
    }
    
    // Payment Sidebar
    const payment = analytics.payment_categorization;
    if (payment && payment.payment_promised) {
        document.getElementById("payment-alert").classList.remove("hidden");
        document.getElementById("payment-details").innerHTML = `
            Expected Date: <strong>${payment.promised_date || "Unknown"}</strong> <br/>
            Method: ${payment.payment_method_mentioned || "Not specified"}
        `;
    } else {
        document.getElementById("payment-alert").classList.add("hidden");
    }

    // PII Sidebar
    const pii = analytics.pii_analysis;
    if (pii && pii.pii_detected && pii.entities) {
        document.getElementById("pii-alert").classList.remove("hidden");
        const list = document.getElementById("pii-list");
        list.innerHTML = "";
        pii.entities.forEach(ent => {
            const li = document.createElement("li");
            li.innerHTML = `<strong>${ent.type}</strong>: ${ent.redacted_value || ent.value} (conf: ${ent.confidence})`;
            list.appendChild(li);
        });
    } else {
        document.getElementById("pii-alert").classList.add("hidden");
    }

    // 3. SOP Validation Detailed Checklist
    document.getElementById("sop-template-badge").innerText = sop.template_used || "Unknown";
    const sopChecklist = document.getElementById("res-sop-checklist");
    sopChecklist.innerHTML = "";
    
    // We iterate over known SOP keys or loop through keys that evaluate to objects with 'status'.
    Object.keys(sop).forEach(key => {
        if(key === 'overall_compliance' || key === 'template_used' || key === 'recommendations' || key === 'custom_checkpoints' || key === 'additional_notes') return;
        
        const checkpoint = sop[key];
        if (checkpoint && checkpoint.status) {
            renderSOPItem(sopChecklist, key.replace(/_/g, " "), checkpoint);
        }
    });

    // Handle array of custom_checkpoints if they exist
    if(sop.custom_checkpoints && Array.isArray(sop.custom_checkpoints)) {
        sop.custom_checkpoints.forEach(cp => {
            renderSOPItem(sopChecklist, cp.checkpoint || "Custom", cp);
        });
    }
    
    // Recommendation block
    if (sop.recommendations && sop.recommendations.length > 0) {
        const recDiv = document.createElement("div");
        recDiv.style.marginTop = "1.5rem";
        recDiv.style.borderTop = "1px solid var(--glass-border)";
        recDiv.style.paddingTop = "1rem";
        recDiv.innerHTML = `<h4>AI Recommendations</h4><ul style="padding-left:1.5rem; color:var(--text-secondary); margin-top:0.5rem">
            ${sop.recommendations.map(r => `<li>${r}</li>`).join('')}
        </ul>`;
        sopChecklist.appendChild(recDiv);
    }

    // 4. Transcription Chat Bubble Render
    const chatContainer = document.getElementById("res-chat");
    chatContainer.innerHTML = "";
    
    if (diarization && diarization.segments && diarization.segments.length > 0) {
        diarization.segments.forEach(seg => {
            const bubble = document.createElement("div");
            const isAgent = (seg.speaker && seg.speaker.toLowerCase() === "agent");
            
            bubble.className = `chat-bubble ${isAgent ? "agent" : "customer"}`;
            
            let badgeClass = "neutral";
            if(seg.sentiment === "positive") badgeClass = "positive";
            if(seg.sentiment === "negative") badgeClass = "negative";

            bubble.innerHTML = `
                <div class="chat-header">
                    <span>${isAgent ? '🎧 Agent' : '👤 Customer'} (${seg.start_time.toFixed(1)}s)</span>
                    <span class="sentiment-badge ${badgeClass}">${seg.sentiment || 'neutral'}</span>
                </div>
                <div class="chat-text">${seg.text}</div>
            `;
            chatContainer.appendChild(bubble);
        });
    } else if (data.transcript) {
        // Fallback to purely un-diarized transcript
        chatContainer.innerHTML = `
            <div style="padding: 1rem; color: var(--text-secondary);">
                <em>Diarization disabled or missing. Raw transcript:</em>
                <br/><br/>
                ${data.transcript}
            </div>
        `;
    } else {
        chatContainer.innerHTML = `<div style="padding: 2rem; text-align: center; opacity: 0.5;">No transcript available.</div>`;
    }

    lucide.createIcons(); // Re-bind icons for dynamic content
}

function renderSOPItem(container, title, checkpointData) {
    const el = document.createElement("div");
    el.className = "sop-item";
    
    let iconName = "help-circle";
    let iconClass = "not_applicable";
    
    if (checkpointData.status === "passed") {
        iconName = "check-circle";
        iconClass = "passed";
    } else if (checkpointData.status === "failed") {
        iconName = "x-circle";
        iconClass = "failed";
    } else if (checkpointData.status === "partial") {
        iconName = "alert-circle";
        iconClass = "partial";
    }

    el.innerHTML = `
        <i data-lucide="${iconName}" class="sop-icon ${iconClass}"></i>
        <div class="sop-content w-full" style="width: 100%">
            <h4>${title}</h4>
            <div style="display:flex; justify-content:space-between; width:100%">
                <p>${checkpointData.details || 'No additional details provided.'}</p>
                <span style="font-size:0.8rem; opacity:0.6">${checkpointData.status.toUpperCase()}</span>
            </div>
            ${checkpointData.evidence ? `<div class="sop-evidence">"${checkpointData.evidence}"</div>` : ''}
        </div>
    `;
    container.appendChild(el);
}
