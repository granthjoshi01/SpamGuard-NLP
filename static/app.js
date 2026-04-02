const form = document.getElementById("spam-form");
const statusText = document.getElementById("form-status");
const submitButton = document.getElementById("submit-button");
const modal = document.getElementById("result-modal");
const closeModalButton = document.getElementById("close-modal");
const languageInput = document.getElementById("language");
const languageSelect = document.getElementById("language-select");
const languageTrigger = document.getElementById("language-trigger");
const languageTriggerLabel = document.getElementById("language-trigger-label");
const languageMenu = document.getElementById("language-menu");
const languageOptions = Array.from(document.querySelectorAll(".select-option"));
const resultPill = document.getElementById("result-pill");
const resultConfidence = document.getElementById("result-confidence");
const resultLanguage = document.getElementById("result-language");
const resultTips = document.getElementById("result-tips");
const resultTranslation = document.getElementById("result-translation");

const languageMap = Object.fromEntries(
  (window.spamGuardLanguages || []).map((item) => [item.code, item.label])
);

function openModal() {
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
}

function closeModal() {
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
}

function setStatus(message, tone = "") {
  statusText.textContent = message;
  statusText.className = `helper-text ${tone}`.trim();
}

function openLanguageMenu() {
  languageMenu.classList.remove("hidden");
  languageTrigger.classList.add("active");
  languageTrigger.setAttribute("aria-expanded", "true");
}

function closeLanguageMenu() {
  languageMenu.classList.add("hidden");
  languageTrigger.classList.remove("active");
  languageTrigger.setAttribute("aria-expanded", "false");
}

function selectLanguage(value, label) {
  languageInput.value = value;
  languageTriggerLabel.textContent = label;

  languageOptions.forEach((option) => {
    const isSelected = option.dataset.value === value;
    option.classList.toggle("selected", isSelected);
    option.setAttribute("aria-selected", String(isSelected));
  });

  closeLanguageMenu();
}

function updateModal(result) {
  const toneClass = result.label.toLowerCase();
  resultPill.textContent = result.label;
  resultPill.className = `result-pill ${toneClass}`;
  resultConfidence.textContent = `${result.confidence}%`;
  resultLanguage.textContent = languageMap[result.language] || result.language;
  resultTips.textContent = result.tips;
  resultTranslation.textContent = result.translated_text;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    language: languageInput.value,
    message: document.getElementById("message").value,
  };

  submitButton.disabled = true;
  setStatus("Checking your SMS with SpamGuard...", "");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Something went wrong while checking the SMS.");
    }

    updateModal(data.result);
    openModal();
    setStatus("Analysis completed successfully.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});

languageTrigger.addEventListener("click", () => {
  if (languageMenu.classList.contains("hidden")) {
    openLanguageMenu();
  } else {
    closeLanguageMenu();
  }
});

languageOptions.forEach((option) => {
  option.addEventListener("click", () => {
    selectLanguage(option.dataset.value, option.dataset.label);
  });
});

closeModalButton.addEventListener("click", closeModal);
modal.addEventListener("click", (event) => {
  if (event.target.dataset.close === "true") {
    closeModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeLanguageMenu();
    closeModal();
  }
});

document.addEventListener("click", (event) => {
  if (!languageSelect.contains(event.target)) {
    closeLanguageMenu();
  }
});
