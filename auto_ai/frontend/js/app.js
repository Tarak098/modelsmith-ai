// State Variables
let currentActiveProjectId = null;
let pollingIntervalId = null;
let uploadedFileObject = null;

// On Page Load Initialization
window.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    loadDashboardStats();
    setupDragAndDrop();
});

// View Navigation Router
function switchView(viewId) {
    // Clear polling if navigating away from running workflow
    if (viewId !== 'workflow' && pollingIntervalId) {
        clearInterval(pollingIntervalId);
        pollingIntervalId = null;
    }

    // Toggle nav active states
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        // Match by text name
        if (item.textContent.trim().toLowerCase().includes(viewId.replace('-', ' '))) {
            item.classList.add('active');
        }
    });

    // Toggle panels
    document.querySelectorAll('.view-panel').forEach(panel => {
        panel.classList.remove('active');
    });

    const activePanel = document.getElementById(`${viewId}-view`);
    if (activePanel) {
        activePanel.classList.add('active');
    }

    // Fetch view-specific records
    if (viewId === 'dashboard') {
        loadDashboardStats();
    } else if (viewId === 'history') {
        loadHistoryList();
    }
}

// Drag & Drop CSV Uploader
function setupDragAndDrop() {
    const dropArea = document.getElementById('drop-area');
    if (!dropArea) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.classList.remove('dragover');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });
}

function handleFileSelected(event) {
    if (event.target.files.length > 0) {
        handleUploadedFile(event.target.files[0]);
    }
}

function handleUploadedFile(file) {
    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
        alert("ModelSmith only accepts CSV files currently.");
        return;
    }
    uploadedFileObject = file;
    document.getElementById('file-label').textContent = `Loaded file: ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
}

// REST API Settings Helpers
async function loadSettings() {
    try {
        const res = await fetch('/api/settings');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('set-key').value = data.gemini_api_key;
            document.getElementById('set-model').value = data.default_model;
            document.getElementById('set-enable-openml').checked = data.enable_openml;
            document.getElementById('set-enable-kaggle').checked = data.enable_kaggle;
            document.getElementById('set-enable-uci').checked = data.enable_uci;
            document.getElementById('set-max-datasets').value = data.max_datasets;
            document.getElementById('set-priority').value = data.repository_priority;
            document.getElementById('set-kaggle-user').value = data.kaggle_username;
            document.getElementById('set-kaggle-key').value = data.kaggle_key;
            document.getElementById('set-threshold').value = data.dataset_score_threshold;
            document.getElementById('set-max-training').value = data.max_training_time;
            document.getElementById('set-max-project').value = data.max_project_time;
            document.getElementById('set-max-retries').value = data.max_retries;
            document.getElementById('set-cv-folds').value = data.cv_folds;
            document.getElementById('set-enable-strategy').checked = data.enable_automl_strategy;
            document.getElementById('set-enable-memory').checked = data.enable_memory_reuse;
            document.getElementById('set-enable-leakage').checked = data.enable_data_leakage_detection;
        }
    } catch (e) {
        console.error("Failed to load settings: ", e);
    }
}

async function saveSettings(event) {
    event.preventDefault();
    const key = document.getElementById('set-key').value;
    const model = document.getElementById('set-model').value;
    const enableOpenML = document.getElementById('set-enable-openml').checked;
    const enableKaggle = document.getElementById('set-enable-kaggle').checked;
    const enableUCI = document.getElementById('set-enable-uci').checked;
    const maxDatasets = parseInt(document.getElementById('set-max-datasets').value) || 10;
    const priority = document.getElementById('set-priority').value;
    const kaggleUser = document.getElementById('set-kaggle-user').value;
    const kaggleKey = document.getElementById('set-kaggle-key').value;
    const threshold = parseFloat(document.getElementById('set-threshold').value) || 0.4;
    const maxTraining = parseInt(document.getElementById('set-max-training').value) || 300;
    const maxProject = parseInt(document.getElementById('set-max-project').value) || 1200;
    const maxRetries = parseInt(document.getElementById('set-max-retries').value) || 5;
    const cvFolds = parseInt(document.getElementById('set-cv-folds').value) || 3;
    const enableStrategy = document.getElementById('set-enable-strategy').checked;
    const enableMemory = document.getElementById('set-enable-memory').checked;
    const enableLeakage = document.getElementById('set-enable-leakage').checked;
    
    try {
        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: jsonStringify({
                gemini_api_key: key,
                default_model: model,
                enable_openml: enableOpenML,
                enable_kaggle: enableKaggle,
                enable_uci: enableUCI,
                max_datasets: maxDatasets,
                repository_priority: priority,
                kaggle_username: kaggleUser,
                kaggle_key: kaggleKey,
                dataset_score_threshold: threshold,
                max_training_time: maxTraining,
                max_project_time: maxProject,
                max_retries: maxRetries,
                cv_folds: cvFolds,
                enable_automl_strategy: enableStrategy,
                enable_memory_reuse: enableMemory,
                enable_data_leakage_detection: enableLeakage
            })
        });
        if (res.ok) {
            alert("Settings updated successfully!");
        }
    } catch (e) {
        alert("Failed to save settings: " + e.message);
    }
}

// REST API Dashboard & History Helpers
async function loadDashboardStats() {
    try {
        const res = await fetch('/api/projects');
        if (!res.ok) return;
        const projects = await res.json();
        
        // Update stats counters
        const total = projects.length;
        const running = projects.filter(p => p.status === 'running').length;
        const completed = projects.filter(p => p.status === 'completed').length;
        const failed = projects.filter(p => p.status === 'failed').length;
        
        document.getElementById('stats-total').textContent = total;
        document.getElementById('stats-running').textContent = running;
        document.getElementById('stats-completed').textContent = completed;
        
        const rate = total > 0 ? Math.round((completed / (completed + failed || 1)) * 100) : 100;
        document.getElementById('stats-rate').textContent = `${rate}%`;
        
        // Populate recent projects list
        const tbody = document.getElementById('recent-projects-list');
        tbody.innerHTML = '';
        
        if (projects.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No projects logged. Start new research!</td></tr>';
            return;
        }
        
        // Display top 5
        projects.slice(0, 5).forEach(p => {
            const dateStr = new Date(p.created_at).toLocaleString();
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${escapeHtml(p.name)}</strong></td>
                <td><span class="badge" style="background:#4f46e5">${escapeHtml(p.category || 'TBD')}</span></td>
                <td><span class="status-badge ${p.status}">${p.status.toUpperCase()}</span></td>
                <td>${dateStr}</td>
                <td>
                    <button class="btn btn-secondary" style="padding:6px 12px; font-size:0.75rem" onclick="viewExistingWorkflow('${p.id}')">Inspect Run</button>
                </td>
            `;
            tbody.appendChild(row);
        });
        
    } catch (e) {
        console.error("Failed to load dashboard stats: ", e);
    }
}

async function loadHistoryList() {
    try {
        const res = await fetch('/api/projects');
        if (!res.ok) return;
        const projects = await res.json();
        
        const tbody = document.getElementById('history-table-list');
        tbody.innerHTML = '';
        
        if (projects.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No projects run yet.</td></tr>';
            return;
        }
        
        projects.forEach(p => {
            const dateStr = new Date(p.created_at).toLocaleString();
            const isFinished = p.status === 'completed';
            const actionBtn = isFinished 
                ? `<button class="btn" style="padding:6px 12px; font-size:0.75rem; background:var(--success-color); box-shadow:none;" onclick="viewReport('${p.id}')">Read Report</button>
                   <a class="btn btn-secondary" style="padding:6px 12px; font-size:0.75rem; display:inline-flex; align-items:center; text-decoration:none;" href="/api/projects/${p.id}/model/download" download>Model 📥</a>` 
                : `<button class="btn btn-secondary" style="padding:6px 12px; font-size:0.75rem" onclick="viewExistingWorkflow('${p.id}')">Telemetry</button>`;
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${escapeHtml(p.name)}</strong></td>
                <td><span class="badge" style="background:#4f46e5">${escapeHtml(p.category || 'TBD')}</span></td>
                <td><span class="status-badge ${p.status}">${p.status.toUpperCase()}</span></td>
                <td>${dateStr}</td>
                <td>
                    <div style="display:flex; gap:8px">
                        ${actionBtn}
                        <button class="btn btn-secondary" style="padding:6px 12px; font-size:0.75rem" onclick="resumeIntoNew('${p.id}', '${escapeJsString(p.name)}')">Improve</button>
                    </div>
                </td>
            `;
            tbody.appendChild(row);
        });
    } catch (e) {
        console.error("Failed to load history list: ", e);
    }
}

// Resume and improve using memory agent lookup trigger
function resumeIntoNew(projectId, name) {
    switchView('new-project');
    document.getElementById('proj-name').value = `Iteration: ${name}`;
    document.getElementById('proj-desc').value = `Improve the model structure for our previous research task. Target model optimization and adjust feature weights.`;
}

// Create New Project Form Submission
async function submitNewProject(event) {
    event.preventDefault();
    
    const name = document.getElementById('proj-name').value;
    const desc = document.getElementById('proj-desc').value;
    
    const formData = new FormData();
    formData.append('name', name);
    formData.append('description', desc);
    if (uploadedFileObject) {
        formData.append('file', uploadedFileObject);
    }
    
    // Disable submission button
    const submitBtn = event.target.querySelector('button[type="submit"]');
    submitBtn.textContent = "Deploying Agents...";
    submitBtn.disabled = true;
    
    try {
        const res = await fetch('/api/projects', {
            method: 'POST',
            body: formData
        });
        
        if (res.ok) {
            const data = await res.json();
            currentActiveProjectId = data.project_id;
            
            // Clear input fields
            document.getElementById('new-project-form').reset();
            document.getElementById('file-label').textContent = "Drag and drop a CSV file here, or click to upload";
            uploadedFileObject = null;
            
            // Show Workflow Telemetry nav item and panel
            document.getElementById('nav-workflow').style.display = 'flex';
            switchView('workflow');
            
            // Reset workflow telemetry display nodes & plots
            resetWorkflowPanel();
            
            // Start Polling Telemetry
            startWorkflowPolling(currentActiveProjectId);
        } else {
            const err = await res.json();
            alert("Execution deployment failed: " + err.detail);
        }
    } catch (e) {
        alert("Server error deploying coordinator: " + e.message);
    } finally {
        submitBtn.textContent = "Deploy AI Agent Team";
        submitBtn.disabled = false;
    }
}

function viewExistingWorkflow(projectId) {
    currentActiveProjectId = projectId;
    document.getElementById('nav-workflow').style.display = 'flex';
    switchView('workflow');
    resetWorkflowPanel();
    startWorkflowPolling(projectId);
}

function resetWorkflowPanel() {
    // Clear nodes
    document.querySelectorAll('.node').forEach(n => {
        n.className = 'node';
    });
    // Reset status label
    document.getElementById('active-agent-display').textContent = 'Orchestrator';
    document.getElementById('project-status-badge').className = 'status-badge running';
    document.getElementById('project-status-badge').textContent = 'RUNNING';
    
    // Reset leaderboard and plots gallery
    document.getElementById('details-section-title').textContent = 'Model Benchmarks Leaderboard';
    document.getElementById('dynamic-details-panel').innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">Waiting for model selection stage to complete...</p>';
    
    document.getElementById('telemetry-plot-gallery').style.display = 'none';
    document.querySelectorAll('.plot-card').forEach(card => card.style.display = 'none');
    document.getElementById('view-report-action-card').style.display = 'none';
    
    // Clear console logs
    document.getElementById('log-terminal').innerHTML = '<div class="log-line log-info">System initialized... waiting for logs...</div>';
}

// Real-Time Run Status Polling
function startWorkflowPolling(projectId) {
    if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
    }
    
    // Perform immediate poll and schedule interval
    pollWorkflowTelemetry(projectId);
    pollingIntervalId = setInterval(() => pollWorkflowTelemetry(projectId), 1500);
}

async function pollWorkflowTelemetry(projectId) {
    try {
        // 1. Fetch Project Main Status
        const projRes = await fetch(`/api/projects/${projectId}`);
        if (!projRes.ok) return;
        const project = await projRes.json();
        
        // Update status badge
        const badge = document.getElementById('project-status-badge');
        badge.className = `status-badge ${project.status}`;
        badge.textContent = project.status.toUpperCase();
        
        // Show/hide conflict resolution panel
        const conflictPanel = document.getElementById('conflict-resolution-panel');
        if (project.status === 'awaiting_feedback') {
            conflictPanel.style.display = 'block';
            document.getElementById('conflict-message-display').textContent = project.error_message || "A task type mismatch conflict was detected. Please select the correct task type below to resolve & resume.";
        } else {
            conflictPanel.style.display = 'none';
        }
        
        // 2. Fetch Tasks Statuses to Color Nodes
        const tasksRes = await fetch(`/api/projects/${projectId}/tasks`);
        if (tasksRes.ok) {
            const tasks = await tasksRes.json();
            
            let activeAgent = "Orchestrator";
            
            tasks.forEach(t => {
                const node = document.getElementById(`node-${t.agent_name}`);
                if (node) {
                    node.className = `node ${t.status}`;
                    if (t.status === 'running') {
                        activeAgent = formatAgentName(t.agent_name);
                    }
                }
                
                // If model_selector has completed, display leaderboard details
                if (t.agent_name === 'model_selector' && t.status === 'completed' && t.output_data) {
                    renderLeaderboardDetails(t.output_data);
                }
                
                // If eda_agent completed, enable charts
                if (t.agent_name === 'eda_agent' && t.status === 'completed') {
                    showPlotCard('heatmap', `/runs/${projectId}/plots/correlation_heatmap.png`);
                    showPlotCard('target', `/runs/${projectId}/plots/target_distribution.png`);
                }
                
                // If evaluator completed, display evaluations
                if (t.agent_name === 'evaluator' && t.status === 'completed') {
                    const isClassification = project.category === 'classification';
                    const plotFile = isClassification ? 'confusion_matrix.png' : 'residual_plot.png';
                    document.getElementById('gallery-header-metric').textContent = isClassification ? 'Confusion Matrix' : 'Residual Analysis Plot';
                    showPlotCard('metric', `/runs/${projectId}/plots/${plotFile}`);
                }
                
                // If explainability completed, display feature importance
                if (t.agent_name === 'explainability' && t.status === 'completed') {
                    showPlotCard('importance', `/runs/${projectId}/plots/feature_importance.png`);
                }
            });
            
            document.getElementById('active-agent-display').textContent = activeAgent;
        }
        
        // 3. Fetch Telemetry Logs
        const logsRes = await fetch(`/api/projects/${projectId}/logs`);
        if (logsRes.ok) {
            const logs = await logsRes.json();
            const terminal = document.getElementById('log-terminal');
            terminal.innerHTML = '';
            
            logs.forEach(l => {
                const logLine = document.createElement('div');
                logLine.className = `log-line log-${l.level.toLowerCase()}`;
                
                const timeStr = new Date(l.timestamp).toLocaleTimeString();
                logLine.textContent = `[${timeStr}] [${l.agent_name.toUpperCase()}] ${l.message}`;
                terminal.appendChild(logLine);
            });
            
            // Auto scroll log terminal to bottom
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        // 4. Check if finished
        if (project.status === 'completed' || project.status === 'failed') {
            clearInterval(pollingIntervalId);
            pollingIntervalId = null;
            
            if (project.status === 'completed') {
                const actionCard = document.getElementById('view-report-action-card');
                actionCard.style.display = 'block';
                document.getElementById('btn-view-finished-report').onclick = () => viewReport(projectId);
                const downloadBtn = document.getElementById('btn-download-built-model');
                if (downloadBtn) {
                    downloadBtn.href = `/api/projects/${projectId}/model/download`;
                }
            } else {
                alert("Agents reported failure: " + project.error_message);
            }
        }
        
    } catch (e) {
        console.error("Polling error: ", e);
    }
}

// Helper methods to dynamically render model selection leaderboard
function renderLeaderboardDetails(data) {
    const title = document.getElementById('details-section-title');
    title.textContent = `Candidate Models Leaderboard (Sorted by ${data.comparison_metric.toUpperCase()})`;
    
    const panel = document.getElementById('dynamic-details-panel');
    let rowsHtml = '';
    
    // Sort leaderboard desc
    const sorted = [...data.leaderboard].sort((a, b) => b.comparison_score - a.comparison_score);
    
    sorted.forEach((row, i) => {
        const isBest = row.model_name === data.best_model_name;
        const metricsStr = Object.entries(row.metrics).map(([k, v]) => `${k.replace('_', ' ')}: ${v.toFixed(3)}`).join(', ');
        rowsHtml += `
            <tr style="${isBest ? 'background: rgba(16, 185, 129, 0.08); font-weight: bold;' : ''}">
                <td>${i+1}</td>
                <td>${row.model_name} ${isBest ? '&nbsp;🏆' : ''}</td>
                <td style="font-size:0.8rem; color:${isBest ? 'var(--success-color)' : 'var(--text-muted)'}">${metricsStr}</td>
            </tr>
        `;
    });
    
    panel.innerHTML = `
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Algorithm</th>
                    <th>Validation Scores</th>
                </tr>
            </thead>
            <tbody>
                ${rowsHtml}
            </tbody>
        </table>
        <p style="margin-top:12px; font-size:0.8rem; color:var(--text-muted)">Selected best estimator: <strong>${data.best_model_name}</strong>.</p>
    `;
}

function showPlotCard(cardId, src) {
    document.getElementById('telemetry-plot-gallery').style.display = 'grid';
    const card = document.getElementById(`gallery-card-${cardId}`);
    const img = document.getElementById(`gallery-img-${cardId}`);
    
    // Cache buster to prevent browser displaying cached previous runs
    img.src = `${src}?cb=${Date.now()}`;
    card.style.display = 'block';
}

function formatAgentName(name) {
    return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

// Serve HTML reports in iframe
function viewReport(projectId) {
    switchView('reports');
    const iframe = document.getElementById('report-pane');
    iframe.src = `/api/projects/${projectId}/report`;
}

// String escape utils
function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function escapeJsString(str) {
    if (!str) return '';
    return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
}

function jsonStringify(obj) {
    return JSON.stringify(obj);
}

async function submitResolveConflict(event) {
    event.preventDefault();
    if (!currentActiveProjectId) return;
    
    const selectEl = document.getElementById('conflict-resolve-select');
    const selectedTaskType = selectEl.value;
    
    try {
        const res = await fetch(`/api/projects/${currentActiveProjectId}/resolve_conflict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_type: selectedTaskType })
        });
        if (res.ok) {
            document.getElementById('conflict-resolution-panel').style.display = 'none';
            // Immediate poller update
            pollWorkflowTelemetry(currentActiveProjectId);
        } else {
            const err = await res.json();
            alert("Failed to resolve conflict: " + (err.detail || "Unknown error"));
        }
    } catch (e) {
        alert("Failed to resolve conflict: " + e.message);
    }
}
