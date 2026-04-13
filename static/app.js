const form = document.getElementById("prediction-form");
const statusText = document.getElementById("status-text");
const output = document.getElementById("prediction-output");
const predictedClassEl = document.getElementById("predicted-class");
const probabilityList = document.getElementById("probability-list");

const fieldIds = [
  "area",
  "perimeter",
  "majorAxisLength",
  "minorAxisLength",
  "aspectRation",
  "eccentricity",
  "convexArea",
  "equivDiameter",
  "extent",
  "solidity",
  "roundness",
  "compactness",
  "shapeFactor1",
  "shapeFactor2",
  "shapeFactor3",
  "shapeFactor4",
];

const classNames = [
  "BARBUNYA",
  "BOMBAY",
  "CALI",
  "DERMASON",
  "HOROZ",
  "SEKER",
  "SIRA",
];

function readFeatures() {
  return fieldIds.map((id) => Number(document.getElementById(id).value));
}

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = "";

  probabilities.forEach((value, idx) => {
    const item = document.createElement("div");
    item.className = "prob-item";
    item.innerHTML = `
      <span>${classNames[idx] || `Class ${idx}`}</span>
      <strong>% ${(value * 100).toFixed(2)}</strong>
    `;
    probabilityList.appendChild(item);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const features = readFeatures();

  if (features.some((v) => Number.isNaN(v))) {
    statusText.textContent = "Please enter valid numeric values for all fields.";
    output.classList.add("hidden");
    return;
  }

  statusText.textContent = "Calculating prediction...";
  output.classList.add("hidden");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });

    const data = await response.json();

    if (!response.ok) {
      statusText.textContent = data.detail || "An error occurred during prediction.";
      output.classList.add("hidden");
      return;
    }

    statusText.textContent = "Prediction completed successfully.";
    predictedClassEl.textContent = data.predicted_class;
    renderProbabilities(data.probabilities);
    output.classList.remove("hidden");
  } catch (error) {
    statusText.textContent = "An error occurred while connecting to the server.";
    output.classList.add("hidden");
  }
});
