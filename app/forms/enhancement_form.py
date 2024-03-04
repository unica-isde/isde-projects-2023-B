from fastapi import Request
from typing import List


class EnhancementForm():

    def __init__(self, request: Request) -> None:
        self.request: Request = request
        self.errors: List = []
        self.color: float
        self.brightness: float
        self.contrast: float
        self.sharpness: float
        self.image_id: str

    async def load_data(self):
        form = await self.request.form()
        self.color = float(form.get("color"))
        self.image_id = form.get("image_id")
        self.brightness = float(form.get("brightness"))
        self.contrast = float(form.get("contrast"))
        self.sharpness = float(form.get("sharpness"))

    def is_valid(self):
        if not self.image_id or not isinstance(self.image_id, str):
            self.errors.append("A valid image id is required")
        if self.color is None or not isinstance(self.color, float):
            self.errors.append("A valid color is required")
        if self.brightness is None or not isinstance(self.brightness, float):
            self.errors.append("A valid brightness is required")
        if self.contrast is None or not isinstance(self.contrast, float):
            self.errors.append("A valid contrast is required")
        if self.sharpness is None or not isinstance(self.sharpness, float):
            self.errors.append("A valid sharpness is required")

        if not self.errors:
            return True
        return False
