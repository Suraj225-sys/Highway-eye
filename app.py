import time
from base64 import b64encode
from pprint import pprint
from dash import Input, Output, html
import cv2
import dash
import dash_player
import dash_bootstrap_components as dbc
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
from PIL import ImageColor, Image
import plotly.express as px
import tensorflow as tf
import tensorflow_hub as hub


def Header(name, app):
    logo = html.Img(
        src=app.get_asset_url("HornbillDrones_Logo.jpeg"), style={"float": "centre", "height": 70}
    )
    title = html.H2(name, style={"margin-left": 5, "text-align": "centre","font-size": "30px"})
    logo_text = html.Span("Login Details", style={"margin-centre": "50px", "color": "black", "font-size": "30px"})

    link = html.A([logo_text], href="https://www.hornbilldrones.com/", style={"color": "black"})

    return dbc.Row([dbc.Col(logo, md=5), dbc.Col(title, md=5), dbc.Col(link, md=2)])




def add_editable_box(
    fig, x0, y0, x1, y1, name=None, color=None, opacity=1, group=None, text=None
):
    fig.add_shape(
        editable=True,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        line_color=color,
        opacity=opacity,
        line_width=3,
        name=name,
    )


# Load colors and detector
colors = list(ImageColor.colormap.values())

module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.load(module_handle).signatures["default"]


# Start app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
server = app.server

controls = [
    dbc.Select(
        id="scene",
        options=[{"label": f"Scene #{i}", "value": i} for i in range(1, 4)],
        value=1,
    ),
    dbc.Button(
        "Detect Frame", id="detect-frame", n_clicks=0, color="primary"
    ),
    dbc.Button(
        "Classify Vehicle", id="classify-vehicle", n_clicks=0, color="primary"
    ),
    html.A(
        dbc.Button("Download", n_clicks=0, color="info", outline=True),
        download="annotations.csv",
        id="download",
    ),


]

video = dbc.Card(
    [
        dbc.CardBody(
            dash_player.DashPlayer(
                id="video", width="100%", height="auto", controls=True
            )
        )
    ]
)

graph_detection = dbc.Card(
    [
        dbc.CardBody(
            dcc.Graph(
                id="graph-detection",
                config={"modeBarButtonsToAdd": ["drawrect"]},
                style={"height": "calc(50vh - 100px)"},
            )
        )
    ]
)

radioitems = html.Div(
    [
        dbc.Label("Choose one"),
        dbc.RadioItems(
            options=[
                {"label": "Car", "value": 1},
                {"label": "Jeep", "value": 2},
                {"label": "Van Private", "value": 3},
                {"label": "Taxi (Yellow board)", "value": 4},
                {"label": "Local Shared Taxi/Chakda", "value": 5},
                {"label": "Mini Bus", "value": 6},
                {"label": "School/Institution Bus", "value": 7},
                {"label": "Bus (Govt)", "value": 8},
                {"label": "Bus (Pvt)", "value": 9},
                {"label": "3 Axle Buses (Govt & Pvt)", "value": 10},
                {"label": "JCB/HCM", "value": 11},
                {"label": "Auto (Goods)", "value": 12},
                {"label": "Auto (Passenger)", "value": 13},
                {"label": "2 Wheeler", "value": 14},
                {"label": "Tractor", "value": 15},
                {"label": "Cycle", "value": 16},
                {"label": "Cycle Rickshaw", "value": 17},
                {"label": "Animal Drawn", "value": 18},
            ],
            value="1",
            id="radioitems-input",
        ),
    ]
)

@app.callback(
    Output("radioitems-input", "value"),
    [Input("radioitems-input", "value")]
)
def handle_radio_input(value):
    print(f"The selected label is {value}")

motorized_labels = ["Car", "Jeep", "Van Private", "Taxi (Yellow board)", "Local Shared Taxi/Chakda", "Mini Bus", "School/Institution Bus", "Bus (Govt)", "Bus (Pvt)", "3 Axle Buses (Govt & Pvt)", "JCB/HCM", "Auto (Goods)", "Auto (Passenger)"]
non_motorized_labels = ["2 Wheeler", "Tractor", "Cycle", "Cycle Rickshaw", "Animal Drawn"]

table_data = [
    {"Motorized": label, "Non Motorized": ""}
    for label in motorized_labels
] + [
    {"Motorized": "", "Non Motorized": label}
    for label in non_motorized_labels
]

table_columns = [
    {"name": "Motorized", "id": "Motorized"},
    {"name": "Non Motorized", "id": "Non Motorized"},
]


classification_text = html.Span("NHI Classification of Vehicles", style={"margin-right": "50px", "color": "black", "font-size": "30px"})

link = html.A([classification_text], href="https://fastag.brokerage-free.in/article/fastag/fastag-vehicle-classification-by-npci-nhai-ihmcl​", style={"color": "black"})

inputs = html.Div(
    [
        dbc.Form([radioitems]),
        html.P(id="radioitems"),
    ]
)

record_table = dbc.Card(
    dash_table.DataTable(
        id="record-table",
        editable=True,
        columns=[
            {"name": i, "id": i}
            for i in [
                "scene",
                "time",
                "order",
                "object",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
            ]
        ],
        data=[],
        page_size=10,
    ),
    body=True,
)

vehicle_table = dbc.Card(
    [
        dash_table.DataTable(
            id="vehicle-table",
            editable=True,
            columns=[
                {"name": "Time", "id": "time"},
                {"name": "Car", "id": "1"},
                {"name": "Jeep", "id": "2"},
                {"name": "Van Private", "id": "3"},
                {"name": "Taxi", "id": "4"},
                {"name": "Bus ", "id": "5"},
                {"name": "JCB/HCM", "id": "6"},
                {"name": "2 Wheeler", "id": "7"},
                {"name": "Tractor", "id": "8"},
                {"name": "Cycle", "id": "9"},
                {"name": "Rickshaw", "id": "10"},
                {"name": "Animal Drawn", "id": "11"},
            ],
            data=[],
            page_size=10,
        )
    ],
    body=True,
)

classify_table = dbc.Card(
    [
        dash_table.DataTable(
            id="classification-table",
            columns=table_columns,
            data=table_data,
            style_table={"width": "50%"},
            style_cell={"textAlign": "center"},
        ),
    ],
    body=True,
)

img_src="E:\HBD\dash_repo\dash-sample-apps\apps\dash-video-detection\assets\classification_grouping.png"

app.layout = dbc.Container(
    [
        Header("Hornbill Drones Highway Eye", app),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        video,
                        html.Br(),
                        link,
                        html.Br(),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        graph_detection,
                        html.Br(),
                        dbc.Card(dbc.Row([dbc.Col(c) for c in controls]), body=True),
                        html.Br(),
                    ],
                    md=7
                ),
                
            ]
        ),
        dbc.Row(
            [   #img_src="E:\HBD\dash_repo\dash-sample-apps\apps\dash-video-detection\assets\classification_grouping.png"
                dbc.Col(
                    [   #img_src="E:\HBD\dash_repo\dash-sample-apps\apps\dash-video-detection\assets\classification_grouping.png",
                        html.Img(src=app.get_asset_url("classification_grouping.jpg"), style={"width": "55%"}),
                        html.Br(),
                    ], md=8
                    ),
                dbc.Col([inputs, html.Br(),], md=4),
            ]
        ),
        dcc.Store(id="store-figure"),
        # dcc.Location(id='url'),
    ],
    fluid=True,
)


@app.callback(Output("video", "url"), [Input("scene", "value")])
def update_scene(i):
    print(app.get_asset_url(f"scene_{i}.mov"))
    return app.get_asset_url(f"scene_{i}.mov")


@app.callback(Output("download", "href"), [Input("record-table", "data")])
def update_download_href(data):
    df = pd.DataFrame.from_records(data)
    df_b64 = b64encode(df.to_csv(index=False).encode())

    return "data:text/csv;base64," + df_b64.decode()


@app.callback(
    Output("record-table", "data"),
    [Input("graph-detection", "relayoutData")],
    [
        State("graph-detection", "figure"),
        State("record-table", "data"),
        State("video", "currentTime"),
        State("scene", "value"),
    ],
)
def update_table(relayout_data, figure, table_data, curr_time, scene):
    if relayout_data is None or figure is None:
        return dash.no_update

    keys = list(relayout_data.keys())
    shapes = figure["layout"]["shapes"]

    if len(keys) == 0:
        return dash.no_update
    elif "shapes" in keys:
        shapes = relayout_data["shapes"]
        i = len(shapes) - 1

    elif "shapes[" in keys[0]:
        i = int(keys[0].replace("shapes[", "").split("].")[0])
    else:
        return dash.no_update

    if i >= len(shapes):
        return dash.no_update

    filtered_table_data = [
        row
        for row in table_data
        if not (
            row["order"] == i
            and row["time"] == round(curr_time, 6)
            and row["scene"] == scene
        )
    ]

    new_shape = shapes[i]
    new = {
        "time": round(curr_time, 6),
        "scene": scene,
        "object": new_shape.get("name", "New"),
        "order": i,
        "xmin": round(new_shape["x0"], 1),
        "xmax": round(new_shape["x1"], 1),
        "ymin": round(new_shape["y0"], 1),
        "ymax": round(new_shape["y1"], 1),
        

    }

    filtered_table_data.append(new)

    return filtered_table_data


@app.callback(
    Output("graph-detection", "figure"),
    [Input("store-figure", "data"), Input("graph-detection", "relayoutData")],
)
def store_to_graph(data, relayout_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update

    prop_id = ctx.triggered[0]["prop_id"]

    if prop_id == "store-figure.data":
        return data

    if "shapes" in relayout_data:
        data["layout"]["shapes"] = relayout_data.get("shapes")
        return data
    else:
        return dash.no_update


@app.callback(
    Output("store-figure", "data"),
    [Input("detect-frame", "n_clicks")],
    [State("scene", "value"), State("video", "currentTime")],
)
def show_time(n_clicks, scene, ms):
    if ms is None or scene is None:
        return dash.no_update

    t0 = time.time()

    cap = cv2.VideoCapture(f"./data/scene-{scene}.mov")
    cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * ms)

    ret, frame = cap.read()
    img = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(img, (512, 512))
    image_tensor = tf.image.convert_image_dtype(resized, tf.float32)[tf.newaxis, ...]
    result = detector(image_tensor)

    boxes = result["detection_boxes"].numpy()
    scores = result["detection_scores"].numpy()
    labels = result["detection_class_entities"].numpy()
    class_ids = result["detection_class_labels"].numpy()

    # Start build figure
    im = Image.fromarray(img)
    fig = px.imshow(im, binary_format="jpg")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0), uirevision=n_clicks
    )

    for i in range(min(10, boxes.shape[0])):
        class_id = scores[i].argmax()
        label = labels[i].decode("ascii")
        confidence = scores[i].max()
        # ymin, xmin, ymax, xmax
        y0, x0, y1, x1 = boxes[i]
        x0 *= im.size[0]
        x1 *= im.size[0]
        y0 *= im.size[1]
        y1 *= im.size[1]

        color = colors[class_ids[i] % len(colors)]

        text = f"{label}: {int(confidence*100)}%"
        print(color)
        print(confidence)
        if confidence > 0.15:
            add_editable_box(
                fig, x0, y0, x1, y1, group=label, name=label, color=color, text=text
            )

    print(f"Detected in {time.time() - t0:.2f}s.")
    #fig = cv2.rotate(fig, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return fig

if __name__ == "__main__":
    app.run_server(debug=False)