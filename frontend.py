import json
import os
from uuid import uuid4

from dash import Input, Output, html
from dotenv import load_dotenv

from src.app_layout import app, clustering_models, dim_reduction_models, mlex_components
from src.callbacks.display import (  # noqa: F401
    clear_selections,
    disable_buttons,
    go_to_first_page,
    go_to_last_page,
    go_to_next_page,
    go_to_prev_page,
    show_clusters,
    show_feature_vectors,
    update_data_overview,
    update_heatmap,
    update_project_name,
)
from src.callbacks.execute import (  # noqa: F401
    allow_run_clustering,
    allow_show_clusters,
    allow_show_feature_vectors,
    run_clustering,
    run_latent_space,
)
from src.callbacks.infrastructure_check import (  # noqa: F401
    check_infra_state,
    update_infra_state,
)
from src.callbacks.live_mode import (  # noqa: F401
    live_update_data_project_dict,
    set_buffered_latent_vectors,
    set_live_latent_vectors,
    toggle_controls,
    toggle_pause_button,
    toggle_pause_button_go_live,
    update_data_project_dict,
)

load_dotenv(".env")

# Define directories
READ_DIR = os.getenv("READ_DIR", "data")
WRITE_DIR = os.getenv("WRITE_DIR", "mlex_store")
MODEL_DIR = f"{WRITE_DIR}/models"
READ_DIR_MOUNT = os.getenv("READ_DIR_MOUNT", None)
WRITE_DIR_MOUNT = os.getenv("WRITE_DIR_MOUNT", None)

# Prefect
PREFECT_TAGS = json.loads(os.getenv("PREFECT_TAGS", '["latent-space-explorer"]'))
TIMEZONE = os.getenv("TIMEZONE", "US/Pacific")
FLOW_NAME = os.getenv("FLOW_NAME", "")

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = os.getenv("APP_PORT", "8070")

# Update UI components for autoencoder model selection.
@app.callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": "feature-extraction-jobs",
        },
        "children",
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-list",
            "aio_id": "feature-extraction-jobs",
        },
        "value",
    ),
)
def update_feature_extraction_model_parameters(model_name):
    from src.app_layout import latent_extraction_models
    model = latent_extraction_models[model_name]
    if model["gui_parameters"]:
        item_list = mlex_components.get_parameter_items(
            _id={"type": str(uuid4())}, json_blob=model["gui_parameters"]
        )
        return item_list
    else:
        return html.Div("Model has no parameters")
    

@app.callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": "latent-space-jobs",
        },
        "children",
    ),
    [
        Input(
            {
                "component": "DbcJobManagerAIO",
                "subcomponent": "model-list",
                "aio_id": "latent-space-jobs",
            },
            "value",
        ),
        Input(
            {
                "component": "DbcJobManagerAIO",
                "subcomponent": "run-dropdown",
                "aio_id": "feature-extraction-jobs",
            },
            "options",
        ),
    ],
)
def update_dim_reduction_model_parameters(model_name, feature_extraction_jobs):
    model = dim_reduction_models[model_name]
    
    if model["gui_parameters"]:
        # Check if there are any feature extraction jobs available
        has_feature_extraction_jobs = feature_extraction_jobs is not None and len(feature_extraction_jobs) > 0
        
        # Modify the input type dropdown to disable latent_features if no jobs are available
        for param in model["gui_parameters"]:
            if param["param_key"] == "input_type":
                # If no feature extraction jobs available, adjust options or tooltips
                if not has_feature_extraction_jobs:
                    # Disable the latent features option or add a notice
                    for option in param["options"]:
                        if option["value"] == "latent_features":
                            option["disabled"] = True
                            option["label"] = "Latent Features (Run feature extraction first)"
        
        item_list = mlex_components.get_parameter_items(
            _id={"type": str(uuid4())}, json_blob=model["gui_parameters"]
        )
        
        # If there are no feature extraction jobs, add a notice
        if not has_feature_extraction_jobs and any(param["param_key"] == "input_type" for param in model["gui_parameters"]):
            notice = html.Div(
                "Note: To use latent features, first run a feature extraction job.",
                style={"color": "red", "font-style": "italic", "margin-top": "10px"}
            )
            item_list.children.append(notice)
            
        return item_list
    else:
        return html.Div("Model has no parameters")

@app.callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": "clustering-jobs",
        },
        "children",
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-list",
            "aio_id": "clustering-jobs",
        },
        "value",
    ),
)
def update_clustering_model_parameters(model_name):
    model = clustering_models[model_name]
    if model["gui_parameters"]:
        item_list = mlex_components.get_parameter_items(
            _id={"type": str(uuid4())}, json_blob=model["gui_parameters"]
        )
        return item_list
    else:
        return html.Div("Model has no parameters")


if __name__ == "__main__":
    app.run_server(
        debug=True,
        host=HOST,
        port=PORT,
    )
