import base64
import plotly.graph_objects as go
import streamlit as st

def visualize(image, bboxes, labels):
    """
    Visualizes the image with bounding boxes and labels using Plotly.

    Args:
        image: The input image.
        bboxes (list): A list of bounding boxes in the format [x1, y1, x2, y2].
        labels (list): A list of labels corresponding to each bounding box.

    """
    # Get the width and height of the image
    width, height = image.size

    shapes = []
    annotations = []

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox

        # Convert bounding box coordinates to the format expected by Plotly
        shapes.append(dict(
            type="rect",
            x0=x1,
            y0=height - y2,
            x1=x2,
            y1=height - y1,
            line=dict(color='red', width=4),
        ))

        # Add annotation for the label
        annotations.append(dict(
            x=(x1 + x2) / 2,
            y=height - y1,
            text=label,
            showarrow=False,
            font=dict(size=20, color='white'),
        ))

    fig = go.Figure()

    # Add the image as a layout image
    fig.update_layout(
        images=[dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="stretch"
        )]
    )

    # Set the axis ranges and disable axis labels
    fig.update_xaxes(range=[0, width], showticklabels=False)
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1,
                     range=[0, width], showticklabels=False)

    fig.update_layout(
        height=800,
        updatemenus=[
            dict(
                direction='left',
                pad=dict(r=5, t=5),
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                type="buttons",
                buttons=[
                    dict(label="Original",
                         method="relayout",
                         args=["shapes", []]),
                    dict(label="Detections",
                         method="relayout",
                         args=["shapes", shapes],
                         ),
                ],
            )
        ],
        annotations=annotations,
    )

    st.plotly_chart(fig)

# Example usage:
  # Example bounding boxes

