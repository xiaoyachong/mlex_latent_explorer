import json

from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate

from src.app_layout import USER
from src.utils.data_utils import tiled_results
from src.utils.plot_utils import generate_scatter_data


@callback(
    Output("show-clusters", "value", allow_duplicate=True),
    Output("show-feature-vectors", "value", allow_duplicate=True),
    Output("sidebar", "style"),
    Output("data-overview-card", "style"),
    Output("image-card", "style"),
    Output("go-live", "style"),
    Output("pause-button", "style"),
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_controls(n_clicks):
    """
    Toggle the visibility of the sidebar, data overview card, image card, and go-live button
    """
    if n_clicks is not None and n_clicks % 2 == 1:
        return (
            False,
            False,
            {"display": "none"},
            {"display": "none"},
            {"width": "98vw", "height": "88vh"},
            {
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "white",
                "background-color": "#00313C",
                "border": "0px",
            },
            {
                "display": "flex",
                "font-size": "1.5rem",
                "padding": "5px",
            },
        )
    else:
        return (
            False,
            False,
            {"overflow-y": "scroll", "height": "90vh"},
            {},
            {"height": "67vh"},
            {
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "#00313C",
                "background-color": "white",
                "border": "0px",
            },
            {
                "display": "none",
            },
        )


@callback(
    Output(
        {"base_id": "file-manager", "name": "data-project-dict"},
        "data",
        allow_duplicate=True,
    ),
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def update_data_project_dict(n_clicks):
    if n_clicks is not None:
        return {
            "root_uri": "",
            "data_type": "tiled",
            "datasets": [],
            "project_id": None,
        }
    else:
        raise PreventUpdate


@callback(
    Output(
        {"base_id": "file-manager", "name": "data-project-dict"},
        "data",
        allow_duplicate=True,
    ),
    Input("ws-live", "message"),
    State("go-live", "n_clicks"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def live_update_data_project_dict(message, n_clicks, data_project_dict):
    """
    Update data project dict with the data uri from the live experiment
    """
    if n_clicks is not None and n_clicks % 2 == 1:
        message = json.loads(message["data"])
        flow_id = message["flow_id"]
        project_name = message["project_name"]

        dim_red_uri = f"{USER}/{project_name}/{flow_id}"
        dim_red_data = tiled_results.get_data_by_trimmed_uri(dim_red_uri)
        metadata = dim_red_data.metadata

        root_uri = metadata["io_parameters"]["root_uri"]
        assert len(metadata["io_parameters"]["data_uris"]) == 1
        data_uri = metadata["io_parameters"]["data_uris"][0]
        data_project_dict["root_uri"] = root_uri

        if len(data_project_dict["datasets"]) == 0:
            cum_size = 1
        else:
            cum_size = data_project_dict["datasets"][-1]["cumulative_data_count"] + 1

        data_project_dict["datasets"] += [
            {
                "uri": data_uri,
                "cumulative_data_count": cum_size,
            }
        ]
    return data_project_dict


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Input("ws-live", "message"),
    State("go-live", "n_clicks"),
    State("scatter", "figure"),
    prevent_initial_call=True,
)
def set_live_latent_vectors(message, n_clicks, current_figure):
    # Parse the incoming message
    message = json.loads(message["data"])
    flow_id = message["flow_id"]
    project_name = message["project_name"]

    # Retrieve data from wherever
    dim_red_uri = f"{USER}/{project_name}/{flow_id}"
    dim_red_data = tiled_results.get_data_by_trimmed_uri(dim_red_uri)
    latent_vectors = dim_red_data.read().to_numpy()
    metadata = dim_red_data.metadata
    num_latent_vectors = latent_vectors.shape[0]

    if "customdata" not in current_figure["data"][0]:
        # Generate new scatter data
        return generate_scatter_data(
            latent_vectors, metadata["model_parameters"]["n_components"]
        )

    else:
        current_figure["data"][0]["customdata"].append([num_latent_vectors - 1])
        current_figure["data"][0]["x"].append(int(latent_vectors[:, 0]))
        current_figure["data"][0]["y"].append(int(latent_vectors[:, 1]))
        return current_figure
