from dataclasses import dataclass

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *


@dataclass
class ExporterOutput:
    save_name: str
    save_type: str
    params: Dict[str, Any]


class Exporter(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        save_video: bool = False

    cfg: Config

    def configure(
        self,
        geometry,
        material,
        background: BaseBackground,
    ) -> None:
        @dataclass
        class SubModules:
            geometry
            material
            background: BaseBackground

        self.sub_modules = SubModules(geometry, material, background)

    @property
    def geometry(self):
        return self.sub_modules.geometry

    @property
    def material(self):
        return self.sub_modules.material

    @property
    def background(self) -> BaseBackground:
        return self.sub_modules.background

    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        raise NotImplementedError


@threestudio.register("dummy-exporter")
class DummyExporter(Exporter):
    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        # DummyExporter does not export anything
        return []
