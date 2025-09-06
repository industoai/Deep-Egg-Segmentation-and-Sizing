"""This is a demo for running the egg segmentation and sizing using streamlit library"""

from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import streamlit as st
import pandas as pd
from PIL import Image


from src.egg_segmentation_size.segmentor import EggSegmentorInference


@dataclass
class DemoEggSegmentationSizing:
    """Class for running the egg segmentation and sizing app using Streamlit."""

    image: str = field(init=False)
    scale_factor: float = field(default=11.61)

    def upload_image(self) -> None:
        """Upload an image from the streamlit page"""
        uploaded_file = st.file_uploader(
            "Upload an image or use the default one...", type=["jpg", "png", "jpeg"]
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
            self.image = tmp_file.name
        else:
            self.image = "tests/test_data/sample1.jpg"
        st.image(
            Image.open(self.image),
            caption="Original/Uploaded Image",
            use_container_width=True,
        )
        self.scale_factor = st.number_input(
            "Choose the scale factor based on your camera for volume calculation as `factor=DPI/2.54` "
            "(For [this dataset](https://huggingface.co/datasets/afshin-dini/Egg-Instance-Segmentation) is 11.61):",
            value=11.61,
            step=0.01,
        )

    def process_image(self) -> None:
        """Process the image for the egg segmentation and sizing"""
        if st.button("Segment/Size Eggs"):
            inferer = EggSegmentorInference(
                model_path=Path("./src/egg_segmentation_size/model/egg_segmentor.pt"),
                result_path="",
                scale_factor=self.scale_factor,
            )
            segmentations = inferer.inference(data_path=self.image)

            result_image = inferer.result_images(segmentations)
            st.markdown("<h3>Segmented Results</h3>", unsafe_allow_html=True)
            st.image(
                result_image[0], caption="Segmented Eggs", use_container_width=True
            )

            res = inferer.results_detail(segmentations)

            extracted_data = []
            if res:
                for key, val in res.items():
                    for detection in val:
                        extracted_data.append(
                            {
                                "Image": key,
                                "Type": detection["class"],
                                "Area in pixel": detection["areas in pixel"],
                                "Volume in cm3": detection["volume in cm3"],
                            }
                        )
            extracted_data = pd.DataFrame(extracted_data).round(2)

            st.markdown('<div class="center-container">', unsafe_allow_html=True)
            st.markdown(
                "<h3>Detailed Information of Segmentations</h3>", unsafe_allow_html=True
            )
            st.markdown(
                """
                <style>
                table {
                    width: 100%;
                }
                th, td {
                    text-align: center !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.table(extracted_data)
            st.markdown("</div>", unsafe_allow_html=True)

    def design_page(self) -> None:
        """Design the streamlit page for eg detector and counter"""
        st.title("Egg segmentor and sizer")
        self.upload_image()
        self.process_image()


demo = DemoEggSegmentationSizing()
demo.design_page()
