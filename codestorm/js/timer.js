/**
 * CodeStorm 2026 - Contest Countdown Timer Module
 * Accurate second-by-second countdown with formatting and expiration handling.
 */

export class ContestTimer {
  /**
   * @param {number} durationMinutes Initial contest duration in minutes (e.g., 90)
   * @param {Function} onTick Callback receiving (formattedTime, remainingSeconds)
   * @param {Function} onExpire Callback triggered on 00:00:00
   */
  constructor(durationMinutes = 90, onTick = null, onExpire = null) {
    this.totalSeconds = Math.max(0, durationMinutes * 60);
    this.remainingSeconds = this.totalSeconds;
    this.onTick = onTick;
    this.onExpire = onExpire;
    this.intervalId = null;
    this.isRunning = false;
  }

  /**
   * Starts the countdown timer.
   */
  start() {
    if (this.isRunning) return;
    this.isRunning = true;

    // Trigger initial tick immediately
    this._tick();

    this.intervalId = setInterval(() => {
      this.remainingSeconds--;
      if (this.remainingSeconds <= 0) {
        this.remainingSeconds = 0;
        this._tick();
        this.stop();
        if (typeof this.onExpire === 'function') {
          this.onExpire();
        }
      } else {
        this._tick();
      }
    }, 1000);
  }

  /**
   * Stops/pauses the timer.
   */
  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.isRunning = false;
  }

  /**
   * Returns current formatted string HH:MM:SS
   * @returns {string}
   */
  getFormattedTime() {
    const hours = Math.floor(this.remainingSeconds / 3600);
    const minutes = Math.floor((this.remainingSeconds % 3600) / 60);
    const seconds = this.remainingSeconds % 60;

    const pad = (num) => String(num).padStart(2, '0');
    return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
  }

  _tick() {
    if (typeof this.onTick === 'function') {
      this.onTick(this.getFormattedTime(), this.remainingSeconds);
    }
  }
}
