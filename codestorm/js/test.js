/**
 * CodeStorm 2026 - Main Test Orchestrator (Phase 3 IDE)
 * Connects question sidebar, problem statement view, Monaco editor, run code drawer, submit confirmation modal, and timer.
 */

import { questions } from './questions.js';
import { ContestTimer } from './timer.js';
import { CodeEditorManager } from './editor.js';

class CodeStormTestApp {
  constructor() {
    this.questions = questions;
    this.currentQuestionIndex = 0;
    this.questionStatuses = {
      1: 'Not Attempted',
      2: 'Not Attempted',
      3: 'Not Attempted',
      4: 'Not Attempted',
      5: 'Not Attempted'
    };

    this.timer = null;
    this.editorManager = null;

    // DOM Elements
    this.dom = {
      timerDisplay: document.getElementById('test-timer-display'),
      questionsList: document.getElementById('questions-nav-list'),
      problemTitle: document.getElementById('problem-title'),
      problemDifficulty: document.getElementById('problem-difficulty'),
      problemStatement: document.getElementById('problem-statement-body'),
      inputFormat: document.getElementById('problem-input-format'),
      outputFormat: document.getElementById('problem-output-format'),
      constraints: document.getElementById('problem-constraints'),
      sampleCasesContainer: document.getElementById('problem-sample-cases'),
      languageSelect: document.getElementById('editor-language-select'),
      runCodeBtn: document.getElementById('btn-run-code'),
      submitBtn: document.getElementById('btn-submit-test'),
      runOutputDrawer: document.getElementById('run-output-drawer'),
      drawerCloseBtn: document.getElementById('btn-close-drawer'),
      drawerContent: document.getElementById('drawer-output-content'),
      submitModal: document.getElementById('submit-confirm-modal'),
      cancelSubmitBtn: document.getElementById('btn-cancel-submit'),
      confirmSubmitBtn: document.getElementById('btn-confirm-submit'),
      editorContainer: document.getElementById('monaco-editor-container')
    };
  }

  /**
   * Initializes the entire test application.
   */
  async init() {
    this._initTimer();
    this._initEditor();
    this._renderSidebar();
    this._renderCurrentProblem();
    this._bindEvents();
  }

  /**
   * Initializes the 90-minute countdown timer.
   */
  _initTimer() {
    this.timer = new ContestTimer(
      90,
      (formattedTime) => {
        if (this.dom.timerDisplay) {
          this.dom.timerDisplay.textContent = formattedTime;
        }
      },
      () => {
        this._handleTimerExpiry();
      }
    );
    this.timer.start();
  }

  /**
   * Initializes Monaco editor and attaches code modification listeners.
   */
  async _initEditor() {
    if (!this.dom.editorContainer) return;

    this.editorManager = new CodeEditorManager(
      this.dom.editorContainer,
      (questionId) => {
        // Mark question as Attempted when user edits code
        if (this.questionStatuses[questionId] !== 'Attempted') {
          this.questionStatuses[questionId] = 'Attempted';
          this._updateSidebarItemStatus(questionId);
        }
      }
    );

    await this.editorManager.init();
  }

  /**
   * Renders the 5 questions in the left sidebar.
   */
  _renderSidebar() {
    if (!this.dom.questionsList) return;

    this.dom.questionsList.innerHTML = this.questions.map((q, idx) => {
      const isActive = idx === this.currentQuestionIndex;
      const status = this.questionStatuses[q.id];
      const isAttempted = status === 'Attempted';

      return `
        <button 
          type="button"
          class="question-nav-item ${isActive ? 'active' : ''} ${isAttempted ? 'is-attempted' : ''}" 
          data-index="${idx}"
          data-id="${q.id}"
          id="nav-q-${q.id}"
          aria-current="${isActive ? 'true' : 'false'}"
        >
          <div class="q-nav-header">
            <span class="q-nav-badge">Q${q.id}</span>
            <span class="q-nav-title">${q.title}</span>
          </div>
          <div class="q-nav-footer">
            <span class="q-status-tag ${isAttempted ? 'tag-attempted' : 'tag-unattempted'}">
              ${status}
            </span>
          </div>
        </button>
      `;
    }).join('');

    // Attach click listeners to sidebar items
    const buttons = this.dom.questionsList.querySelectorAll('.question-nav-item');
    buttons.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const index = parseInt(btn.getAttribute('data-index'), 10);
        this.selectQuestion(index);
      });
    });
  }

  /**
   * Updates status badge for a specific question in the sidebar.
   */
  _updateSidebarItemStatus(questionId) {
    const btn = document.getElementById(`nav-q-${questionId}`);
    if (!btn) return;

    const status = this.questionStatuses[questionId];
    const isAttempted = status === 'Attempted';

    btn.classList.toggle('is-attempted', isAttempted);
    const tag = btn.querySelector('.q-status-tag');
    if (tag) {
      tag.className = `q-status-tag ${isAttempted ? 'tag-attempted' : 'tag-unattempted'}`;
      tag.textContent = status;
    }
  }

  /**
   * Switches to a specific question index ($0 \dots 4$).
   */
  selectQuestion(index) {
    if (index < 0 || index >= this.questions.length) return;
    this.currentQuestionIndex = index;

    // Update sidebar active classes
    const buttons = this.dom.questionsList.querySelectorAll('.question-nav-item');
    buttons.forEach((btn, idx) => {
      const isActive = idx === index;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-current', isActive ? 'true' : 'false');
    });

    this._renderCurrentProblem();

    // Switch Monaco code editor state
    const currentQ = this.questions[index];
    if (this.editorManager) {
      this.editorManager.setQuestion(currentQ.id);
    }

    // Scroll problem panel back to top
    const problemPanel = document.querySelector('.problem-panel');
    if (problemPanel) problemPanel.scrollTop = 0;
  }

  /**
   * Renders full problem description for the active question.
   */
  _renderCurrentProblem() {
    const q = this.questions[this.currentQuestionIndex];
    if (!q) return;

    if (this.dom.problemTitle) this.dom.problemTitle.textContent = `${q.id}. ${q.title}`;

    if (this.dom.problemDifficulty) {
      this.dom.problemDifficulty.textContent = q.difficulty;
      this.dom.problemDifficulty.className = `badge badge-difficulty badge-${q.difficulty.toLowerCase()}`;
    }

    if (this.dom.problemStatement) this.dom.problemStatement.innerHTML = q.statement;
    if (this.dom.inputFormat) this.dom.inputFormat.innerHTML = q.inputFormat;
    if (this.dom.outputFormat) this.dom.outputFormat.innerHTML = q.outputFormat;
    if (this.dom.constraints) this.dom.constraints.innerHTML = q.constraints;

    // Render sample test cases
    if (this.dom.sampleCasesContainer) {
      this.dom.sampleCasesContainer.innerHTML = q.sampleCases.map((sample, idx) => `
        <div class="sample-case-block">
          <div class="sample-case-header">
            <span class="sample-case-title">Example ${idx + 1}</span>
          </div>
          <div class="sample-io-grid">
            <div class="sample-io-column">
              <span class="sample-io-label">Input:</span>
              <pre class="sample-code-box"><code>${sample.input}</code></pre>
            </div>
            <div class="sample-io-column">
              <span class="sample-io-label">Output:</span>
              <pre class="sample-code-box"><code>${sample.output}</code></pre>
            </div>
          </div>
          ${sample.explanation ? `
            <div class="sample-explanation">
              <strong>Explanation:</strong> ${sample.explanation}
            </div>
          ` : ''}
        </div>
      `).join('');
    }
  }

  /**
   * Binds interaction events for language selector, run code, and submit confirmation.
   */
  _bindEvents() {
    // Language dropdown change
    this.dom.languageSelect?.addEventListener('change', (e) => {
      const selectedLang = e.target.value;
      if (this.editorManager) {
        this.editorManager.setLanguage(selectedLang);
      }
    });

    // Run Code button click
    this.dom.runCodeBtn?.addEventListener('click', () => {
      this._handleRunCode();
    });

    // Close Run Output drawer
    this.dom.drawerCloseBtn?.addEventListener('click', () => {
      this.dom.runOutputDrawer?.classList.remove('active');
    });

    // Submit button click -> Show confirmation modal
    this.dom.submitBtn?.addEventListener('click', () => {
      this._openSubmitModal();
    });

    // Cancel submit modal
    this.dom.cancelSubmitBtn?.addEventListener('click', () => {
      this._closeSubmitModal();
    });

    // Confirm submit -> Redirect to submitted.html
    this.dom.confirmSubmitBtn?.addEventListener('click', () => {
      this._handleSubmitContest();
    });

    // Keyboard escape to close modals/drawers
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this._closeSubmitModal();
        this.dom.runOutputDrawer?.classList.remove('active');
      }
    });

    // Window resize handler for Monaco layout
    window.addEventListener('resize', () => {
      this.editorManager?.layout();
    });
  }

  /**
   * Mock execution interaction.
   */
  async _handleRunCode() {
    if (!this.dom.runOutputDrawer || !this.dom.drawerContent) return;

    this.dom.runOutputDrawer.classList.add('active');
    this.dom.drawerContent.innerHTML = `
      <div class="execution-loading">
        <span class="loading-spinner" aria-hidden="true"></span>
        <span>Compiling and executing against sample test cases...</span>
      </div>
    `;

    // Simulate clean async backend API call: POST /api/code/run
    await this._mockRunCodeApi();

    const currentQ = this.questions[this.currentQuestionIndex];
    const testCasesHtml = currentQ.sampleCases.map((sample, idx) => `
      <div class="testcase-result-card passed">
        <div class="tc-header">
          <span class="tc-name">Test Case ${idx + 1}</span>
          <span class="badge badge-success">Passed</span>
        </div>
        <div class="tc-details">
          <div><span class="tc-label">Execution Time:</span> <span class="tc-val">0.0${idx + 1}s</span></div>
          <div><span class="tc-label">Memory:</span> <span class="tc-val">12.4 MB</span></div>
        </div>
      </div>
    `).join('');

    this.dom.drawerContent.innerHTML = `
      <div class="execution-results">
        <div class="exec-summary success">
          <span class="summary-icon">&#x2713;</span>
          <span class="summary-text">All Sample Test Cases Passed</span>
        </div>
        <div class="testcases-grid">
          ${testCasesHtml}
        </div>
      </div>
    `;
  }

  /**
   * API Wrapper stub: POST /api/code/run
   */
  async _mockRunCodeApi() {
    return new Promise((resolve) => setTimeout(resolve, 400));
  }

  /**
   * Opens the Submit confirmation modal.
   */
  _openSubmitModal() {
    if (!this.dom.submitModal) return;

    // Count attempted questions
    const attemptedCount = Object.values(this.questionStatuses).filter((s) => s === 'Attempted').length;
    const countEl = document.getElementById('modal-attempted-count');
    if (countEl) {
      countEl.textContent = `${attemptedCount} of 5`;
    }

    this.dom.submitModal.classList.add('active');
    this.dom.confirmSubmitBtn?.focus();
  }

  /**
   * Closes the Submit confirmation modal.
   */
  _closeSubmitModal() {
    this.dom.submitModal?.classList.remove('active');
    this.dom.submitBtn?.focus();
  }

  /**
   * Handles contest submission and redirection.
   */
  async _handleSubmitContest() {
    // Simulate backend submission API call: POST /api/submission
    await this._mockSubmitApi();
    window.location.href = 'submitted.html';
  }

  /**
   * API Wrapper stub: POST /api/submission
   */
  async _mockSubmitApi() {
    return new Promise((resolve) => setTimeout(resolve, 150));
  }

  /**
   * Handles timer reaching 00:00:00.
   */
  _handleTimerExpiry() {
    const alertModal = document.createElement('div');
    alertModal.className = 'modal-backdrop active';
    alertModal.innerHTML = `
      <div class="modal-card" role="alertdialog" aria-modal="true">
        <div class="modal-header">
          <h2 class="modal-title" style="color: #f87171;">Time Expired</h2>
          <p class="modal-desc">The 90-minute contest limit has ended. Your solutions are being recorded automatically.</p>
        </div>
        <div class="modal-footer" style="justify-content: center;">
          <span class="loading-spinner" aria-hidden="true"></span>
          <span style="font-size: var(--text-sm); color: var(--text-secondary);">Submitting your test...</span>
        </div>
      </div>
    `;
    document.body.appendChild(alertModal);

    setTimeout(() => {
      window.location.href = 'submitted.html';
    }, 1500);
  }
}

// Auto-boot on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new CodeStormTestApp();
  app.init();
});
