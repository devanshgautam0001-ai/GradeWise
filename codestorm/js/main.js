/**
 * CodeStorm 2026 - Main Interactive Scripts (ES6 Vanilla Module)
 * Phase 2 Scope: Student access flow, mock test data binding, key validation, and navigation.
 */

import { mockTest } from './mock-data.js';

document.addEventListener('DOMContentLoaded', () => {
  initStudentLogin();
  initTestKey();
  initTestInfo();
  initAdminLogin();
});

/**
 * Handles Student Login Form with lightweight inline validation and mock navigation.
 */
function initStudentLogin() {
  const form = document.getElementById('student-login-form');
  if (!form) return;

  const studentIdInput = document.getElementById('student-id');
  const passwordInput = document.getElementById('student-password');
  const studentIdGroup = document.getElementById('student-id-group');
  const passwordGroup = document.getElementById('student-password-group');

  // Clear errors on typing
  studentIdInput?.addEventListener('input', () => {
    studentIdGroup?.classList.remove('has-error');
  });

  passwordInput?.addEventListener('input', () => {
    passwordGroup?.classList.remove('has-error');
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    let hasError = false;

    const studentId = studentIdInput?.value.trim();
    const password = passwordInput?.value.trim();

    if (!studentId) {
      studentIdGroup?.classList.add('has-error');
      hasError = true;
    }

    if (!password) {
      passwordGroup?.classList.add('has-error');
      hasError = true;
    }

    if (hasError) return;

    // Mock successful authentication -> Navigate to Test Key step
    window.location.href = 'test-key.html';
  });
}

/**
 * Handles Test Key validation against isolated mock test data.
 */
function initTestKey() {
  const form = document.getElementById('test-key-form');
  if (!form) return;

  const keyInput = document.getElementById('test-key-input');
  const keyGroup = document.getElementById('test-key-group');
  const feedbackEl = document.getElementById('test-key-feedback');

  keyInput?.addEventListener('input', () => {
    keyGroup?.classList.remove('has-error');
    if (feedbackEl) {
      feedbackEl.className = 'status-feedback';
      feedbackEl.innerHTML = '';
    }
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const enteredKey = keyInput?.value.trim().toUpperCase();

    if (!enteredKey || enteredKey !== mockTest.key.toUpperCase()) {
      keyGroup?.classList.add('has-error');
      if (feedbackEl) {
        feedbackEl.className = 'status-feedback active status-error';
        feedbackEl.setAttribute('role', 'alert');
        feedbackEl.innerHTML = `
          <strong>Invalid Test Key</strong><br>
          Please check the key provided by the organizers.
        `;
      }
      return;
    }

    // Valid mock test key -> Navigate to Test Info
    window.location.href = 'test-info.html';
  });
}

/**
 * Dynamically populates Test Information view from centralized mock data.
 */
function initTestInfo() {
  const infoContainer = document.getElementById('test-info-view');
  if (!infoContainer) return;

  const nameEl = document.getElementById('info-test-name');
  const titleEl = document.getElementById('info-test-title');
  const durationEl = document.getElementById('info-test-duration');
  const questionsEl = document.getElementById('info-test-questions');
  const languagesEl = document.getElementById('info-test-languages');
  const statusEl = document.getElementById('info-test-status');

  if (nameEl) nameEl.textContent = mockTest.name;
  if (titleEl) titleEl.textContent = mockTest.title;
  if (durationEl) durationEl.textContent = `${mockTest.duration} Minutes`;
  if (questionsEl) questionsEl.textContent = `${mockTest.questions}`;
  if (languagesEl) languagesEl.textContent = mockTest.languages.join(' / ');
  if (statusEl) statusEl.textContent = mockTest.status;
}

/**
 * Handles Admin Login Form (Phase 1 Preview Mode).
 */
function initAdminLogin() {
  const adminForm = document.getElementById('admin-login-form');
  if (!adminForm) return;

  adminForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const feedbackEl = document.getElementById('admin-auth-feedback');
    const adminId = document.getElementById('admin-id')?.value.trim();

    if (feedbackEl) {
      feedbackEl.className = 'status-feedback active status-admin';
      feedbackEl.setAttribute('role', 'status');
      feedbackEl.innerHTML = `
        <strong>[ PHASE 1 PREVIEW ]</strong><br>
        Admin ID <code>${escapeHtml(adminId || 'N/A')}</code> received.<br>
        Admin portal dashboard, problem creation, and test monitoring will be activated in Phase 5.
      `;
    }
  });
}

/**
 * Basic HTML escaping helper for safe feedback string injection.
 * @param {string} str 
 * @returns {string}
 */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
