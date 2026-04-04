document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audio-file');
    const dropZone = document.getElementById('drop-zone');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const results = document.getElementById('results');

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults (e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        updateFileText(files[0].name);
    });

    fileInput.addEventListener('change', (e) => {
        if(e.target.files.length > 0) {
            updateFileText(e.target.files[0].name);
        }
    });

    function updateFileText(name) {
        dropZone.querySelector('h3').textContent = name;
        dropZone.querySelector('p').textContent = 'Ready to analyze';
    }

    // Analysis trigger
    analyzeBtn.addEventListener('click', async () => {
        if (fileInput.files.length === 0) {
            alert('Please select an MP3 file first!');
            return;
        }

        const file = fileInput.files[0];
        const apiKey = document.getElementById('api-key').value;
        const language = document.getElementById('language').value;

        // Convert to Base64
        const reader = new FileReader();
        reader.readAsDataURL(file);
        
        loader.classList.remove('hidden');
        results.classList.add('hidden');
        analyzeBtn.disabled = true;

        reader.onload = async () => {
            const base64String = reader.result.split(',')[1];
            
            try {
                // Determine API endpoint (use absolute or relative based on environment)
                const apiUrl = '/api/call-analytics';
                
                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'x-api-key': apiKey
                    },
                    body: JSON.stringify({
                        language: language,
                        audioFormat: 'mp3',
                        audioBase64: base64String
                    })
                });

                const data = await response.json();
                
                if(!response.ok) {
                    throw new Error(data.detail || data.message || "Unknown error");
                }

                renderResults(data);
                
            } catch (error) {
                alert('API Error: ' + error.message);
            } finally {
                loader.classList.add('hidden');
                analyzeBtn.disabled = false;
            }
        };
    });

    function renderResults(data) {
        // Score logic
        const score = data.sop_validation.complianceScore * 100;
        document.getElementById('score-text').textContent = score + '%';
        const circle = document.querySelector('.score-circle');
        circle.style.background = `conic-gradient(var(--success) ${score}%, rgba(255,255,255,0.1) 0%)`;

        // Status Badge
        const badge = document.getElementById('adherence-status');
        badge.textContent = data.sop_validation.adherenceStatus;
        badge.className = 'status-badge ' + (data.sop_validation.adherenceStatus === 'FOLLOWED' ? 'success' : 'danger');

        // SOP List
        const sopItems = ['greeting', 'identification', 'problemStatement', 'solutionOffering', 'closing'];
        const listEl = document.getElementById('sop-list');
        listEl.innerHTML = '';
        sopItems.forEach(item => {
            const li = document.createElement('li');
            const isChecked = data.sop_validation[item];
            const icon = isChecked ? 'fa-check-circle' : 'fa-times-circle';
            li.innerHTML = `<span>${item.charAt(0).toUpperCase() + item.slice(1).replace(/([A-Z])/g, ' $1')}</span> <i class="fa-solid ${icon}"></i>`;
            listEl.appendChild(li);
        });

        // Analytics
        document.getElementById('sentiment').textContent = data.analytics.sentiment;
        document.getElementById('payment').textContent = data.analytics.paymentPreference;
        
        // Advanced Metrics safely fallback
        const adv = data.advanced_metrics || {};
        const agentTalk = adv.agent_talk_percent ? `${adv.agent_talk_percent}% Agent` : 'N/A';
        document.getElementById('talk-ratio').textContent = agentTalk;
        document.getElementById('sentiment-shift').textContent = adv.sentiment_shift || 'Static';

        // Transcript & Summary
        document.getElementById('transcript-box').innerHTML = data.redacted_transcript ? data.redacted_transcript.replace(/\n/g, '<br>') : data.transcript.replace(/\n/g, '<br>');
        document.getElementById('summary-box').textContent = data.summary;

        results.classList.remove('hidden');
    }
});
