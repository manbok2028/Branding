"""브랜드 이름부터 로고까지 한 번에 만들어 주는 프로그램.

이 파일의 전체 흐름은 다음과 같습니다.

1. brief.json에서 만들고 싶은 브랜드 정보를 읽습니다.
2. AI에게 브랜드 이름과 슬로건을 물어봅니다.
3. 선택한 이름을 사용해 브랜드 이야기를 만듭니다.
4. 브랜드와 어울리는 색을 추천받습니다.
5. 추천 색으로 색상표 이미지와 로고 이미지 2개를 만듭니다.
6. 모든 결과를 output 폴더에 저장합니다.

위에서 아래로 천천히 읽으면 프로그램이 일하는 순서를 이해할 수 있습니다.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from openai import OpenAI

# 그래프와 이미지에 한글이 네모로 깨지지 않도록 Windows 한글 폰트를 사용합니다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ====================================================================
# 0. 준비하기
# 여러 기능에서 함께 사용하는 파일 위치와 AI 모델 이름을 정합니다.
# ====================================================================

# main.py가 들어 있는 폴더의 위치입니다.
BASE_DIR = Path(__file__).resolve().parent

# 별도의 경로를 입력하지 않으면 아래 파일과 폴더를 사용합니다.
DEFAULT_BRIEF = BASE_DIR / "brief.json"  # 브랜드 설명이 담긴 파일
DEFAULT_OUTPUT = BASE_DIR / "output"  # 완성된 결과를 담을 폴더

# 글을 만드는 AI와 그림을 만드는 AI의 기본 모델 이름입니다.
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")


def load_env() -> None:
    """비밀번호처럼 중요한 API 키를 .env 파일에서 읽습니다."""
    load_dotenv(BASE_DIR / ".env")

    # 두 가지 이름 중 하나로 키를 적어도 사용할 수 있게 이름을 맞춥니다.
    if not os.getenv("OPENAI_API_KEY") and os.getenv("GPT_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["GPT_API_KEY"]


def get_openai_client() -> OpenAI:
    """AI에게 질문을 보낼 수 있는 연결 도구를 준비합니다."""
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY 또는 GPT_API_KEY가 없습니다. .env 파일에 설정하세요."
        )
    # API 키가 있어야 OpenAI 서버에 질문을 보낼 수 있습니다.
    return OpenAI(api_key=api_key)


def load_brief(path: Path) -> dict[str, Any]:
    """brief.json을 읽고 꼭 필요한 내용이 들어 있는지 확인합니다."""
    if not path.exists():
        raise FileNotFoundError(f"브리프 파일을 찾을 수 없습니다: {path}")

    # JSON 글자를 파이썬이 사용하기 편한 딕셔너리로 바꿉니다.
    brief = json.loads(path.read_text(encoding="utf-8"))

    # 업종, 고객, 키워드 중 하나라도 비어 있으면 이유를 알려줍니다.
    missing = [key for key in ("industry", "target", "keywords") if not brief.get(key)]
    if missing:
        raise ValueError(f"brief.json의 필수 항목이 비어 있습니다: {', '.join(missing)}")
    if not isinstance(brief["keywords"], list):
        raise ValueError("brief.json의 keywords는 문자열 배열이어야 합니다.")
    return brief


def ensure_output_dir(output_dir: str | Path) -> Path:
    """결과 폴더가 없으면 새로 만듭니다."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ask_llm(prompt: str) -> str:
    """AI에게 질문을 보내고 답변 글자를 돌려받습니다."""
    response = get_openai_client().responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=prompt,
    )
    return response.output_text


def parse_json_response(text: str) -> Any:
    """AI가 보낸 JSON 글자를 파이썬에서 사용할 수 있게 바꿉니다."""
    cleaned = text.strip()

    # AI가 답을 ``` 표시로 감쌌다면 그 표시만 떼어냅니다.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(f"AI 응답을 JSON으로 해석할 수 없습니다: {error}") from error


# ====================================================================
# 1. 브랜드 이름과 슬로건 만들기
# brief.json의 내용을 질문에 넣어 AI에게 아이디어를 부탁합니다.
# ====================================================================

def generate_naming(brief: dict) -> list:
    """AI에게 브랜드 이름 후보 3~5개와 이름의 뜻을 부탁합니다."""

    # f"""...""" 안의 중괄호에는 brief.json의 실제 내용이 들어갑니다.
    prompt = f"""
다음 브랜드 브리프를 참고해 브랜드명 후보 3~5개를 제안해줘.
브리프에 기존 브랜드명이 있으면 첫 번째 후보로 유지하고, 그 의미와 서비스 연관성을 설명해줘.

기존 브랜드명: {brief.get('brand_name', '미정')}
업종: {brief['industry']}
타겟: {brief['target']}
키워드: {', '.join(brief['keywords'])}
톤앤매너: {brief.get('tone', '자유롭게')}

아래 JSON 배열 형식으로만 답해. 다른 설명이나 코드 블록은 추가하지 마.
[{{"name": "브랜드명", "meaning": "의미 또는 유래"}}]
"""
    # 먼저 AI에게 질문하고, 받은 JSON 답변을 목록으로 바꿉니다.
    return parse_json_response(ask_llm(prompt))


def generate_slogan(brief: dict) -> list:
    """AI에게 브랜드를 짧게 표현하는 슬로건 3개를 부탁합니다."""
    prompt = f"""
다음 브랜드 브리프를 참고해 슬로건 3개를 제안해줘.

브랜드명: {brief.get('brand_name', '미정')}
업종: {brief['industry']}
타겟: {brief['target']}
키워드: {', '.join(brief['keywords'])}
톤앤매너: {brief.get('tone', '자유롭게')}

아래 JSON 배열 형식으로만 답해. 다른 설명이나 코드 블록은 추가하지 마.
["슬로건 1", "슬로건 2", "슬로건 3"]
"""
    return parse_json_response(ask_llm(prompt))


# ====================================================================
# 2. 브랜드 이야기 만들기
# 앞 단계에서 만든 브랜드 이름을 사용해 탄생 배경과 철학을 만듭니다.
# ====================================================================

def generate_story(brief: dict, brand_name: str) -> str:
    """브랜드 정보와 이름을 AI에게 보내 브랜드 이야기를 만듭니다."""

    # 같은 뜻의 키 이름이 들어와도 사용할 수 있도록 차례로 찾아봅니다.
    category = brief.get("category", brief.get("industry", ""))
    target = brief.get("target_audience", brief.get("target", ""))
    core_values = brief.get("core_values", brief.get("keywords", []))
    values = ", ".join(core_values)
    tone = brief.get("tone", "브랜드 목적에 맞는 신뢰감 있는 톤")
    description = brief.get("description", "")

    print("[브랜드 스토리 생성 시작]")
    print(f"브랜드명: {brand_name}")
    print(f"카테고리: {category}")
    print("-" * 40)

    prompt = f"""
당신은 다양한 산업의 브랜드 전략과 카피라이팅에 능숙한 전문 작가입니다.
아래 브랜드 정보를 바탕으로 신뢰감 있고 공감할 수 있는 브랜드 스토리를 작성해주세요.

브랜드명: {brand_name}
카테고리: {category}
타겟 고객: {target}
핵심 가치: {values}
톤앤매너: {tone}
서비스 설명: {description}

조건:
- 200~300자 분량
- 브랜드의 탄생 배경과 철학을 담을 것
- 타겟 고객의 공감을 이끌어낼 것
- {tone} 톤으로 작성할 것
- 서비스가 제공하지 않는 전문 자문이나 결과를 보장하는 표현은 쓰지 말 것
"""

    # system은 AI의 역할, user는 AI에게 부탁할 내용을 뜻합니다.
    response = get_openai_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "당신은 브랜드 스토리 전문 작가입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    # 여러 답변 중 첫 번째 답변에서 앞뒤의 빈칸을 지웁니다.
    story = (response.choices[0].message.content or "").strip()
    print("[생성된 브랜드 스토리]")
    print(story)
    print("-" * 40)
    return story


# ====================================================================
# 3. 브랜드 색과 색상표 이미지 만들기
# AI가 추천한 색상 코드를 실제 PNG 그림으로 바꿉니다.
# ====================================================================

def normalize_hex(hex_code: str) -> str:
    """색상 코드를 '#RRGGBB' 모양으로 통일합니다."""
    hex_code = hex_code.strip().upper()
    if not hex_code.startswith("#"):
        hex_code = "#" + hex_code
    if len(hex_code) != 7:
        raise ValueError(f"잘못된 HEX 코드 형식입니다: {hex_code}")
    return hex_code


def generate_colors(
    brief: dict,
    client: OpenAI,
    model: str = DEFAULT_TEXT_MODEL,
) -> dict:
    """AI에게 대표 색 1개와 보조 색 2~3개를 추천받습니다."""

    # system_prompt에는 AI가 어떤 전문가처럼 행동할지 적습니다.
    system_prompt = (
        "너는 브랜드 비주얼 디렉터다. 사용자가 제공한 브랜드 브리프를 바탕으로 "
        "브랜드에 어울리는 컬러 팔레트를 제안하라. 반드시 JSON만 출력하라."
    )
    user_prompt = f"""
다음 브랜드 브리프를 바탕으로 컬러 팔레트를 추천해줘.

[브랜드 브리프]
- 업종: {brief.get('industry', '')}
- 타겟: {brief.get('target', '')}
- 키워드: {', '.join(brief.get('keywords', []))}
- 톤앤매너: {brief.get('tone', '')}
- 설명: {brief.get('description', brief.get('notes', ''))}

[요구사항]
1. 메인 컬러 1개, 서브 컬러 2~3개를 추천해줘.
2. 각 컬러는 이름(name)과 HEX 코드(hex)를 포함해줘.
3. 브리프의 업종, 핵심 가치, 톤앤매너를 색상에 구체적으로 반영해줘.
4. 서비스의 신뢰성, 명료함, 접근성을 고려하고 불안이나 위기감을 과도하게 자극하는 색은 피해야 해.
5. 반드시 아래 JSON 형식으로만 출력해줘.

{{
  "main": {{"name": "컬러명", "hex": "#RRGGBB"}},
  "sub": [
    {{"name": "컬러명", "hex": "#RRGGBB"}},
    {{"name": "컬러명", "hex": "#RRGGBB"}}
  ]
}}
"""
    # response_format을 사용해 답변을 JSON 모양으로 받습니다.
    response = client.chat.completions.create(
        model=model,
        temperature=0.8,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    colors = json.loads(content)

    # AI 답변에 대표 색과 보조 색이 모두 있는지 확인합니다.
    if "main" not in colors or "sub" not in colors:
        raise ValueError("컬러 응답 형식이 올바르지 않습니다.")
    colors["main"]["hex"] = normalize_hex(colors["main"]["hex"])
    if not isinstance(colors["sub"], list) or len(colors["sub"]) < 2:
        raise ValueError("서브 컬러는 최소 2개 이상이어야 합니다.")
    for item in colors["sub"]:
        item["hex"] = normalize_hex(item["hex"])
    return colors


def create_color_palette(colors: dict, output_dir: str | Path) -> str:
    """추천받은 색을 나란히 그려 color_palette.png로 저장합니다."""

    # 완성될 이미지의 전체 저장 위치입니다.
    output_path = ensure_output_dir(output_dir) / "color_palette.png"
    all_colors = [colors["main"]] + colors["sub"]
    color_count = len(all_colors)

    fig_width = max(10, 2.6 * color_count)
    # 도화지(fig)와 그 위에 그림을 그릴 공간(ax)을 만듭니다.
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    ax.set_xlim(0, color_count)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 색상 하나마다 색칠된 네모와 이름을 하나씩 그립니다.
    for index, color in enumerate(all_colors):
        rect = Rectangle(
            (index, 0.35), 1, 0.45, facecolor=color["hex"], edgecolor="white"
        )
        ax.add_patch(rect)
        role = "MAIN" if index == 0 else f"SUB {index}"
        label = f"{role}\n{color['name']}\n{color['hex']}"
        ax.text(index + 0.5, 0.18, label, ha="center", va="center", fontsize=10)

    ax.text(
        0, 0.95, "Brand Color Palette", fontsize=14, fontweight="bold", ha="left", va="top"
    )
    plt.tight_layout()
    # 완성된 도화지를 PNG 파일로 저장하고 메모리에서 닫습니다.
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# ====================================================================
# 4. 로고 이미지 만들기
# 브랜드 이름, 소개, 색을 모아 그림 AI가 이해할 수 있는 질문을 만듭니다.
# ====================================================================

def build_logo_prompt(brief: dict, colors: dict, brand_name: str) -> str:
    """로고를 어떻게 그릴지 설명하는 긴 질문을 만듭니다."""

    # 대표 색, 보조 색, 키워드를 질문에 넣기 좋은 글자로 합칩니다.
    main_color = colors["main"]["hex"]
    sub_colors = ", ".join(color["hex"] for color in colors["sub"])
    keywords = ", ".join(brief.get("keywords", []))
    return f"""
Create a clear, trustworthy, modern logo concept for a Korean web service named
"{brand_name}".

Brand context:
- Industry: {brief.get('industry', '')}
- Target: {brief.get('target', '')}
- Keywords: {keywords}
- Tone: {brief.get('tone', '')}
- Brand description: {brief.get('description', brief.get('notes', ''))}

Design direction:
- Translate the supplied brand brief into a calm, reassuring, and approachable identity.
- Emphasize clarity, trust, guidance, and a constructive next step.
- Avoid imagery that guarantees debt cancellation, legal outcomes, or tax relief.
- Avoid intimidating government emblems, alarmist warning symbols, and an overly corporate style.
- Use {main_color} as the dominant color.
- Supporting colors may include {sub_colors}.
- Use a minimal, soft, modern style suitable for a web/app service.
- Include the Korean brand name "{brand_name}" in a readable way if possible.
- Use a white or very light background.
""".strip()


def generate_logos(
    brief: dict,
    colors: dict,
    client: OpenAI,
    output_dir: str | Path,
    brand_name: str,
    num_images: int = 2,
    model: str = DEFAULT_IMAGE_MODEL,
) -> list[str]:
    """그림 AI에게 요청해 서로 다른 로고 시안을 저장합니다."""
    output_path = ensure_output_dir(output_dir)
    saved_paths: list[str] = []
    base_prompt = build_logo_prompt(brief, colors, brand_name)
    # 같은 브랜드라도 강조점을 달리해 여러 시안을 만듭니다.
    variations = [
        "Focus on a reset-to-next-step concept using restrained, abstract directional forms.",
        "Focus on trustworthy guidance, organized information, and calm forward movement.",
        "Focus on clean web-service branding and a minimal, accessible icon style.",
    ]

    # num_images가 2라면 아래 작업을 두 번 반복합니다.
    for index in range(num_images):
        variation = variations[index] if index < len(variations) else "Create a distinct concept."
        result = client.images.generate(
            model=model,
            prompt=f"{base_prompt}\n\nVariation {index + 1}: {variation}",
            size="1024x1024",
        )
        # 인터넷으로 받은 이미지 글자를 다시 실제 이미지 데이터로 바꿉니다.
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # 첫 번째는 logo_01.png, 두 번째는 logo_02.png로 저장합니다.
        file_path = output_path / f"logo_{index + 1:02d}.png"
        file_path.write_bytes(image_bytes)
        saved_paths.append(str(file_path))
    return saved_paths


# ====================================================================
# 5. 모든 기능을 순서대로 실행하기
# 여기서 위의 작은 함수들을 차례대로 불러 하나의 결과로 합칩니다.
# ====================================================================

def build_brand_identity(brief: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """이름부터 로고까지 만들고 모든 결과를 한곳에 모읍니다."""

    # 결과를 저장할 폴더와 AI 연결 도구를 먼저 준비합니다.
    output_dir = ensure_output_dir(output_dir)
    client = get_openai_client()

    print("[1/5] 브랜드 네이밍 생성 중...")
    names = generate_naming(brief)
    # brief.json에 확정 브랜드명이 있으면 이를 우선 사용합니다.
    # 확정 이름이 없을 때만 첫 번째 AI 후보 또는 일반적인 임시 이름을 사용합니다.
    brand_name = str(brief.get("brand_name", "브랜드"))
    if not brief.get("brand_name") and names:
        first_name = names[0]
        brand_name = first_name.get("name", brand_name) if isinstance(first_name, dict) else str(first_name)

    print("[2/5] 슬로건 생성 중...")
    slogans = generate_slogan(brief)

    print("[3/5] 브랜드 스토리 생성 중...")
    story = generate_story(brief, brand_name)

    print("[4/5] 컬러 팔레트 생성 중...")
    colors = generate_colors(brief, client)
    palette_path = create_color_palette(colors, output_dir)

    print("[5/5] 로고 시안 생성 중...")
    logo_paths = generate_logos(
        brief=brief,
        colors=colors,
        client=client,
        output_dir=output_dir,
        brand_name=brand_name,
        num_images=2,
    )

    # 지금까지 만든 결과를 하나의 큰 딕셔너리에 담습니다.
    result: dict[str, Any] = {
        "brief": brief,
        "selected_brand_name": brand_name,
        "names": names,
        "slogans": slogans,
        "story": story,
        "colors": colors,
        "palette_path": palette_path,
        "logo_paths": logo_paths,
    }
    # 큰 딕셔너리를 사람이 읽기 쉬운 JSON 파일로 저장합니다.
    result_path = output_dir / "brand_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    """실행할 때 사용자가 입력한 파일 경로 선택 사항을 읽습니다."""

    # 예: python main.py --brief brief.json --output output
    parser = argparse.ArgumentParser(description="브랜드 아이덴티티 생성기")
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF, help="브리프 파일 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="결과 폴더 경로")
    return parser.parse_args()


def main() -> int:
    """프로그램을 시작하고, 문제가 생기면 쉬운 오류 문장을 보여줍니다."""
    args = parse_args()
    try:
        build_brand_identity(load_brief(args.brief), args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[오류] {error}", file=sys.stderr)
        return 1
    print(f"완료: {args.output / 'brand_result.json'}")
    return 0


# 이 파일을 직접 실행했을 때만 main()을 시작합니다.
# 다른 파일에서 가져다 쓸 때는 자동으로 실행되지 않습니다.
if __name__ == "__main__":
    raise SystemExit(main())
