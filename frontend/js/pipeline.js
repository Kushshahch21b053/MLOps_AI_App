document.addEventListener("DOMContentLoaded", function () {
  const pipelineStagesContainer = document.querySelector(".pipeline-stages");

  // Define pipeline stages (just names, no runtime or lastRun)
  const pipelineStages = [
    { name: "Data Processing" },
    { name: "Feature Engineering" },
    { name: "Model Training" },
    { name: "Model Evaluation" },
    { name: "Model Deployment" },
  ];

  // Generate pipeline visualization
  generatePipelineStages(pipelineStages);

  function generatePipelineStages(stages) {
    stages.forEach((stage, index) => {
      // Create stage element
      const stageEl = document.createElement("div");
      stageEl.className = "pipeline-stage";
      stageEl.innerHTML = `<h4>${stage.name}</h4>`;

      // Create wrapper with connector
      const wrapper = document.createElement("div");
      wrapper.className = "stage-wrapper";
      wrapper.appendChild(stageEl);

      // Add arrow connector if not the last stage
      if (index < stages.length - 1) {
        const arrow = document.createElement("div");
        arrow.className = "stage-connector";
        arrow.innerHTML = "→";
        wrapper.appendChild(arrow);
      }

      pipelineStagesContainer.appendChild(wrapper);
    });
  }
});
