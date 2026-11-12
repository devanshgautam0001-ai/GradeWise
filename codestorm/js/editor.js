/**
 * CodeStorm 2026 - Monaco Code Editor Controller
 * Manages Monaco editor instance, language modes, starter templates, and per-question code preservation.
 */

import { questions, starterCodeTemplates } from './questions.js';

export class CodeEditorManager {
  /**
   * @param {HTMLElement} containerElement DOM container for the editor
   * @param {Function} onCodeModified Callback when code is edited (passes questionId)
   */
  constructor(containerElement, onCodeModified = null) {
    this.container = containerElement;
    this.onCodeModified = onCodeModified;
    this.editorInstance = null;
    this.activeQuestionId = 1;
    this.activeLanguage = 'cpp'; // 'cpp' | 'java' | 'python'
    
    // In-memory code state: { [questionId]: { cpp: string, java: string, python: string } }
    this.codeState = {};
    this.isProgrammaticChange = false;

    this._initCodeState();
  }

  /**
   * Pre-populates default starter codes for all questions.
   */
  _initCodeState() {
    questions.forEach((q) => {
      this.codeState[q.id] = {
        cpp: q.starterCode?.cpp || starterCodeTemplates.cpp,
        java: q.starterCode?.java || starterCodeTemplates.java,
        python: q.starterCode?.python || starterCodeTemplates.python
      };
    });
  }

  /**
   * Initializes the Monaco Editor via CDN AMD loader.
   * @returns {Promise<void>}
   */
  async init() {
    return new Promise((resolve) => {
      // If Monaco is already loaded globally
      if (window.monaco) {
        this._createEditor();
        resolve();
        return;
      }

      // Check if require.js loader is present
      if (window.require) {
        window.require.config({
          paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' }
        });
        window.require(['vs/editor/editor.main'], () => {
          this._createEditor();
          resolve();
        });
        return;
      }

      // Dynamically load Monaco loader
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/loader.js';
      script.onload = () => {
        window.require.config({
          paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' }
        });
        window.require(['vs/editor/editor.main'], () => {
          this._createEditor();
          resolve();
        });
      };
      script.onerror = () => {
        // Fallback for offline / blocked CDN environments: Create styled textarea
        this._createFallbackEditor();
        resolve();
      };
      document.head.appendChild(script);
    });
  }

  /**
   * Creates the Monaco Editor instance.
   */
  _createEditor() {
    const initialCode = this.getCode(this.activeQuestionId, this.activeLanguage);

    this.editorInstance = window.monaco.editor.create(this.container, {
      value: initialCode,
      language: this._mapLanguageToMonaco(this.activeLanguage),
      theme: 'vs',
      fontSize: 14,
      fontFamily: '"JetBrains Mono", "SF Mono", "Fira Code", Menlo, Consolas, monospace',
      lineNumbers: 'on',
      roundedSelection: false,
      scrollBeyondLastLine: false,
      automaticLayout: true,
      tabSize: 4,
      minimap: { enabled: false },
      bracketPairColorization: { enabled: true },
      padding: { top: 12, bottom: 12 },
      suggestOnTriggerCharacters: true
    });

    // Listen to user edits
    this.editorInstance.onDidChangeModelContent(() => {
      if (this.isProgrammaticChange) return;

      const currentCode = this.editorInstance.getValue();
      this.codeState[this.activeQuestionId][this.activeLanguage] = currentCode;

      if (typeof this.onCodeModified === 'function') {
        this.onCodeModified(this.activeQuestionId, this.activeLanguage, currentCode);
      }
    });
  }

  /**
   * Fallback textarea editor if external Monaco CDN is unavailable.
   */
  _createFallbackEditor() {
    this.container.innerHTML = `
      <textarea 
        id="fallback-code-input"
        class="fallback-editor" 
        spellcheck="false"
        aria-label="Code Editor"
      ></textarea>
    `;
    const textarea = this.container.querySelector('#fallback-code-input');
    textarea.value = this.getCode(this.activeQuestionId, this.activeLanguage);

    textarea.addEventListener('input', () => {
      const currentCode = textarea.value;
      this.codeState[this.activeQuestionId][this.activeLanguage] = currentCode;
      if (typeof this.onCodeModified === 'function') {
        this.onCodeModified(this.activeQuestionId, this.activeLanguage, currentCode);
      }
    });

    this.editorInstance = {
      getValue: () => textarea.value,
      setValue: (val) => { textarea.value = val; },
      layout: () => {}
    };
  }

  /**
   * Maps friendly language names to Monaco language IDs.
   */
  _mapLanguageToMonaco(lang) {
    switch (lang) {
      case 'cpp': return 'cpp';
      case 'java': return 'java';
      case 'python': return 'python';
      default: return 'cpp';
    }
  }

  /**
   * Switches the active question and loads its saved code.
   * @param {number} questionId 
   */
  setQuestion(questionId) {
    // Save current editor content before switching
    if (this.editorInstance) {
      this.codeState[this.activeQuestionId][this.activeLanguage] = this.editorInstance.getValue();
    }

    this.activeQuestionId = questionId;
    const targetCode = this.getCode(questionId, this.activeLanguage);

    this.isProgrammaticChange = true;
    if (this.editorInstance) {
      this.editorInstance.setValue(targetCode);
    }
    this.isProgrammaticChange = false;
  }

  /**
   * Switches the programming language mode.
   * @param {string} lang 'cpp' | 'java' | 'python'
   */
  setLanguage(lang) {
    if (this.activeLanguage === lang) return;

    // Save current language code
    if (this.editorInstance) {
      this.codeState[this.activeQuestionId][this.activeLanguage] = this.editorInstance.getValue();
    }

    this.activeLanguage = lang;
    const targetCode = this.getCode(this.activeQuestionId, lang);

    this.isProgrammaticChange = true;
    if (this.editorInstance) {
      if (window.monaco && this.editorInstance.getModel) {
        window.monaco.editor.setModelLanguage(
          this.editorInstance.getModel(),
          this._mapLanguageToMonaco(lang)
        );
      }
      this.editorInstance.setValue(targetCode);
    }
    this.isProgrammaticChange = false;
  }

  /**
   * Retrieves code for a specific question & language.
   */
  getCode(questionId, lang = this.activeLanguage) {
    return this.codeState[questionId]?.[lang] || '';
  }

  /**
   * Returns current editor code.
   */
  getCurrentCode() {
    return this.editorInstance ? this.editorInstance.getValue() : '';
  }

  /**
   * Triggers editor layout update.
   */
  layout() {
    if (this.editorInstance && typeof this.editorInstance.layout === 'function') {
      this.editorInstance.layout();
    }
  }
}
