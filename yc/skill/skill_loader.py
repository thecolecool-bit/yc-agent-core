import re
from pathlib import Path
from typing import Dict, Optional, List
import yaml
from yc.common.exceptions import AppError
from yc.schemas.skill import Skill


class SkillLoader:
    def __init__(self, basepath: str):
        self.basepath = basepath
        self._skills: Dict[str, Skill] = {}
        self._scan_skills()

    def _scan_skills(self):
        base_dir = Path(self.basepath)
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
        for skill_dir in base_dir.iterdir():
            if skill_dir.is_file():
                continue
            skill_md = next(skill_dir.glob("SKILL.md"), None)
            if skill_md is None:
                continue
            metadata = self._parse_metadata(skill_md)
            if not metadata:
                continue
            self._skills[metadata['name']] = Skill(name=metadata['name'], description=metadata['description'],
                                                   path=skill_dir.__str__())

    @staticmethod
    def _parse_metadata(path: Path) -> Optional[Dict]:
        try:
            content = path.read_text(encoding='utf-8')
        except Exception as exc:
            raise AppError("解析skill元数据失败", detail={"error": str(exc)})
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not match:
            return None
        yaml_str = match.group(1)
        try:
            metadata = yaml.safe_load(yaml_str) or {}
        except yaml.YAMLError:
            return None
        # 验证必需字段
        if "name" not in metadata or "description" not in metadata:
            return None
        return metadata

    def load_skill(self, skill_name: str) -> Optional[Skill]:
        if skill_name not in self._skills:
            return None
        skill = self._skills[skill_name]
        if skill.content is None:
            skill_md = next(Path(skill.path).glob("SKILL.md"), None)
            if skill_md is None:
                return None
            with skill_md.open('r', encoding='utf-8') as f:
                skill.content = f.read()
            if not skill.content:
                return None
            match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', skill.content, re.DOTALL)
            if not match:
                return None
            metainfo, body = match.groups()
            skill.content = body.strip()
        return skill

    def get_skills_metadata(self) -> List[Skill]:
        return list(self._skills.values())


skill_loader = SkillLoader('skills')
