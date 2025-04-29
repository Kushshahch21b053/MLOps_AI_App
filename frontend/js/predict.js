document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("prediction-form");
  const resultDiv = document.getElementById("prediction-result");
  const errorDiv = document.getElementById("error-message");
  const resetBtn = document.getElementById("reset-btn");
  const sensorInputsContainer = document.querySelector(".sensor-inputs");

  // Generate sensor input fields
  generateSensorInputs();

  // Handle form submission
  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Show loading state
    document.querySelector('.form-actions button[type="submit"]').textContent =
      "Predicting...";
    resultDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");

    try {
      // Collect form data
      const engineId = parseInt(document.getElementById("engine-id").value);
      const settings = [
        parseFloat(document.getElementById("setting1").value),
        parseFloat(document.getElementById("setting2").value),
        parseFloat(document.getElementById("setting3").value),
      ];

      // Collect sensor values
      const sensors = [];
      for (let i = 1; i <= 21; i++) {
        const value = parseFloat(document.getElementById(`sensor${i}`).value);
        sensors.push(value);
      }

      // Create payload
      const payload = {
        engine_id: engineId,
        settings: settings,
        sensors: sensors,
      };

      // Send prediction request
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const data = await response.json();

      // Display results
      document.getElementById("rul-number").textContent =
        data.remaining_useful_life.toFixed(2);
      document.getElementById(
        "confidence-interval"
      ).textContent = `${data.confidence_interval.lower.toFixed(
        2
      )} - ${data.confidence_interval.upper.toFixed(2)} cycles`;
      document.getElementById("prediction-time").textContent = new Date(
        data.prediction_time
      ).toLocaleString();

      resultDiv.classList.remove("hidden");
    } catch (error) {
      console.error("Error:", error);
      errorDiv.classList.remove("hidden");
    } finally {
      // Reset button text
      document.querySelector(
        '.form-actions button[type="submit"]'
      ).textContent = "Predict RUL";
    }
  });

  // Handle reset button
  resetBtn.addEventListener("click", function () {
    form.reset();
    resultDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");
  });

  // Function to generate sensor input fields
  function generateSensorInputs() {
    const sensorLabels = [
      "T2",
      "T24",
      "T30",
      "T50",
      "P2",
      "P15",
      "P30",
      "Nf",
      "Nc",
      "epr",
      "Ps30",
      "phi",
      "NRf",
      "NRc",
      "BPR",
      "farB",
      "htBleed",
      "Nf_dmd",
      "PCNfR_dmd",
      "W31",
      "W32",
    ];

    // Sample values from a typical engine
    const defaultValues = [
      518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61, 554.36, 2388.06, 9046.19,
      1.3, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100, 38.5,
      23.0,
    ];

    // Create grid layout for sensors
    const grid = document.createElement("div");
    grid.className = "sensor-grid";

    for (let i = 0; i < sensorLabels.length; i++) {
      const group = document.createElement("div");
      group.className = "form-group";

      const label = document.createElement("label");
      label.htmlFor = `sensor${i + 1}`;
      label.textContent = `${sensorLabels[i]}:`;

      const input = document.createElement("input");
      input.type = "number";
      input.id = `sensor${i + 1}`;
      input.name = `sensor${i + 1}`;
      input.value = defaultValues[i];
      input.step = "0.01";
      input.required = true;

      group.appendChild(label);
      group.appendChild(input);
      grid.appendChild(group);
    }

    sensorInputsContainer.appendChild(grid);
  }
});
