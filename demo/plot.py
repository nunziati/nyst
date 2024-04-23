import pickle

import pandas as pd
import plotly.express as px

def main():
    with open("output_dict.pkl", "rb") as f:
        output_dict = pickle.load(f)

    df = {"position": {"left": None, "right": None}, "speed": {resolution: {"left": None, "right": None} for resolution in output_dict["speed"]}}
    fig = {"position": {"left": None, "right": None}, "speed": {resolution: {"left": None, "right": None} for resolution in output_dict["speed"]}}

    position_left = output_dict["position"]["left"]
    position_right = output_dict["position"]["right"]

    df["position"]["left"] = pd.DataFrame(position_left, columns=["x", "y"])
    fig["position"]["left"] = px.line(df["position"]["left"], x=df["position"]["left"].index, y=["x", "y"], title="Left Eye Position")
    fig["position"]["left"].update_xaxes(rangeslider_visible=True)
    fig["position"]["left"].write_html(f'position_left.html', auto_open=True)

    df["position"]["right"] = pd.DataFrame(position_right, columns=["x", "y"])
    fig["position"]["right"] = px.line(df["position"]["right"], x=df["position"]["right"].index, y=["x", "y"], title="Right Eye Position")
    fig["position"]["right"].update_xaxes(rangeslider_visible=True)
    fig["position"]["left"].write_html(f'position_right.html', auto_open=True)

    speed_dict = output_dict["speed"]

    for resolution in speed_dict:
        left_speed = speed_dict[resolution]["left"]
        right_speed = speed_dict[resolution]["right"]

        df["speed"][resolution]["left"] = pd.DataFrame(left_speed, columns=["x", "y"])
        fig["speed"][resolution]["left"] = px.line(df["speed"][resolution]["left"], x=df["speed"][resolution]["left"].index, y=["x", "y"], title=f"Left Eye Speed (resolution = {resolution})")
        fig["speed"][resolution]["left"].update_xaxes(rangeslider_visible=True)
        fig["speed"][resolution]["left"].update_yaxes(range=[-250, 250])
        fig["speed"][resolution]["left"].write_html(f'speed_left_{resolution}.html', auto_open=True)

        df["speed"][resolution]["right"] = pd.DataFrame(right_speed, columns=["x", "y"])
        fig["speed"][resolution]["right"] = px.line(df["speed"][resolution]["right"], x=df["speed"][resolution]["right"].index, y=["x", "y"], title=f"Right Eye Speed (resolution = {resolution})")
        fig["speed"][resolution]["right"].update_xaxes(rangeslider_visible=True)
        fig["speed"][resolution]["right"].update_yaxes(range=[-250, 250])
        fig["speed"][resolution]["right"].write_html(f'speed_right_{resolution}.html', auto_open=True)

if __name__ == "__main__":
    main()