import os
import json
import base64
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# 공통 설정
# -----------------------------
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")


# -----------------------------
# 기본 유틸
# -----------------------------
def load_brief(brief_path: str) -> dict:
    """brief.json 파일을 읽어서 dict로 반환"""
    path = Path(brief_path)
    if not path.exists():
        raise FileNotFoundError(f"브리프 파일을 찾을 수 없습니다: {brief_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir(output_dir: str) -> Path:
    """출력 폴더 생성"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_openai_client() -> OpenAI:
    """환경변수에서 API 키를 읽어 OpenAI 클라이언트 생성"""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. "
            ".env 파일 또는 환경변수를 확인하세요."
        )

    return OpenAI(api_key=api_key)


def normalize_hex(hex_code: str) -> str:
    """HEX 코드 정리"""
    hex_code = hex_code.strip().upper()
    if not hex_code.startswith("#"):
        hex_code = "#" + hex_code
    if len(hex_code) != 7:
        raise ValueError(f"잘못된 HEX 코드 형식입니다: {hex_code}")
    return hex_code


# -----------------------------
# 1) GPT로 컬러 팔레트 생성
# -----------------------------
def generate_colors(brief: dict, client: OpenAI, model: str = DEFAULT_TEXT_MODEL) -> dict:
    """
    GPT를 이용해 메인 1개, 서브 2~3개의 컬러 팔레트 생성
    반환 예시:
    {
      "main": {"name": "Soft Green", "hex": "#7CCB5E"},
      "sub": [
        {"name": "Warm Cream", "hex": "#F6E7A1"},
        {"name": "Soft Orange", "hex": "#FFB86C"},
        {"name": "Milky White", "hex": "#FFF8E8"}
      ]
    }
    """
    system_prompt = (
        "너는 브랜드 비주얼 디렉터다. "
        "사용자가 제공한 브랜드 브리프를 바탕으로 브랜드에 어울리는 컬러 팔레트를 제안하라. "
        "반드시 JSON만 출력하라."
    )

    user_prompt = f"""
다음 브랜드 브리프를 바탕으로 컬러 팔레트를 추천해줘.

[브랜드 브리프]
- 업종: {brief.get("industry", "")}
- 타겟: {brief.get("target", "")}
- 키워드: {", ".join(brief.get("keywords", []))}
- 톤앤매너: {brief.get("tone", "")}
- 설명: {brief.get("description", brief.get("notes", ""))}

[요구사항]
1. 메인 컬러 1개, 서브 컬러 2~3개를 추천해줘.
2. 각 컬러는 이름(name)과 HEX 코드(hex)를 포함해줘.
3. 컬러는 너무 차갑고 기술적인 느낌보다, 편안하고 다정하며 부담 없는 분위기를 반영해줘.
4. 음식/생활편의/안심/자취 감성을 고려해줘.
5. 반드시 아래 JSON 형식으로만 출력해줘.

{{
  "main": {{
    "name": "컬러명",
    "hex": "#RRGGBB"
  }},
  "sub": [
    {{
      "name": "컬러명",
      "hex": "#RRGGBB"
    }},
    {{
      "name": "컬러명",
      "hex": "#RRGGBB"
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.8,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    colors = json.loads(content)

    # 최소 검증
    if "main" not in colors or "sub" not in colors:
        raise ValueError("컬러 응답 형식이 올바르지 않습니다.")

    colors["main"]["hex"] = normalize_hex(colors["main"]["hex"])

    if not isinstance(colors["sub"], list) or len(colors["sub"]) < 2:
        raise ValueError("서브 컬러는 최소 2개 이상이어야 합니다.")

    for item in colors["sub"]:
        item["hex"] = normalize_hex(item["hex"])

    return colors


# -----------------------------
# 2) 컬러 팔레트 이미지 생성
# -----------------------------
def create_color_palette(colors: dict, output_dir: str = "./output") -> str:
    """
    colors 정보를 바탕으로 color_palette.png 생성
    반환값: 저장 경로
    """
    output_path = ensure_output_dir(output_dir) / "color_palette.png"

    all_colors = [colors["main"]] + colors["sub"]
    n = len(all_colors)

    fig_width = max(10, 2.6 * n)
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, color in enumerate(all_colors):
        rect = Rectangle((i, 0.35), 1, 0.45, facecolor=color["hex"], edgecolor="white")
        ax.add_patch(rect)

        role = "MAIN" if i == 0 else f"SUB {i}"
        label = f"{role}\n{color['name']}\n{color['hex']}"
        ax.text(
            i + 0.5,
            0.18,
            label,
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.text(
        0,
        0.95,
        "Brand Color Palette",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


# -----------------------------
# 3) 로고 프롬프트 생성
# -----------------------------
def build_logo_prompt(brief: dict, colors: dict, brand_name: str = "푸트테크") -> str:
    """
    로고 생성용 프롬프트를 조립
    """
    main_color = colors["main"]["hex"]
    sub_colors = ", ".join([c["hex"] for c in colors["sub"]])
    keywords = ", ".join(brief.get("keywords", []))

    prompt = f"""
Create a clean, trustworthy, modern logo concept for a Korean service named "{brand_name}".

Brand context:
- Industry: {brief.get("industry", "")}
- Target: {brief.get("target", "")}
- Keywords: {keywords}
- Tone: {brief.get("tone", "")}
- Brand description: {brief.get("description", brief.get("notes", ""))}

Design direction:
- Follow the supplied industry, target, keywords, tone, and description precisely.
- Make the brand feel reassuring, transparent, clear, and approachable.
- Communicate organized guidance and an actionable next step without promising a guaranteed outcome.
- Avoid a cold, intimidating, bureaucratic, or overly corporate style.
- Emphasize clarity, trust, calm progress, and practical usefulness.
- Create an ownable symbol with a clear idea, not a generic stock icon.
- Use bold, balanced geometry and intentional negative space so the mark remains legible at app-icon size.

Visual hints:
- Use the main color {main_color} as the dominant color.
- Supporting colors may include: {sub_colors}.
- Derive a distinctive motif from the supplied brand context instead of using unrelated stock imagery.
- Prefer a minimal, soft, modern logo.
- Make it suitable for a web/app service.
- Include the exact Korean brand name "{brand_name}" in a clearly readable way.
- Use crisp flat vector-style shapes on a white or very light background.
- Keep strong contrast between the symbol, wordmark, and background.
- Use at most one small accent-color detail.

Strictly avoid:
- gradients, glow, shadows, bevels, embossing, 3D effects, dark vignettes, and photographic textures
- overly thin or fragile strokes
- generic clip-art symbols and decorative visual clutter
- mockup scenes, extra text, and watermarks

Output:
- Generate a polished logo concept image.
"""
    return prompt.strip()


# -----------------------------
# 4) 로고 시안 생성
# -----------------------------
def generate_logos(
    brief: dict,
    colors: dict,
    client: OpenAI,
    output_dir: str = "./output",
    brand_name: str = "푸트테크",
    num_images: int = 2,
    model: str = DEFAULT_IMAGE_MODEL,
) -> list[str]:
    """
    OpenAI 이미지 생성 API로 로고 시안 생성
    반환값: 저장된 파일 경로 리스트
    """
    output_path = ensure_output_dir(output_dir)
    saved_paths = []

    base_prompt = build_logo_prompt(brief, colors, brand_name=brand_name)

    variation_instructions = [
        "Variation 1: create a distinctive monogram-style symbol with a continuous visual flow and premium simplicity.",
        "Variation 2: create a bold emblem-style symbol with optical symmetry, protective shapes, and one restrained accent detail.",
        "Variation 3: create a clean app-service identity with a compact, memorable icon and confident silhouette.",
    ]

    for i in range(num_images):
        extra = variation_instructions[i] if i < len(variation_instructions) else f"Variation {i+1}: create a distinct but consistent logo concept."
        prompt = f"{base_prompt}\n\n{extra}"

        result = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
        )

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        file_path = output_path / f"logo_{i+1:02d}.png"
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        saved_paths.append(str(file_path))

    return saved_paths


# -----------------------------
# 5) 실행 함수
# -----------------------------
def run_visual_pipeline(
    brief_path: str,
    output_dir: str = "./output",
    brand_name: str = "푸트테크",
) -> dict:
    """
    visual.py 전체 파이프라인 실행
    - brief.json 읽기
    - 컬러 생성
    - 팔레트 저장
    - 로고 저장
    반환값: 결과 정보 dict
    """
    client = get_openai_client()
    brief = load_brief(brief_path)

    print("[4/5] 컬러 팔레트 생성 중...")
    colors = generate_colors(brief, client=client)

    main_hex = colors["main"]["hex"]
    main_name = colors["main"]["name"]
    sub_hex_list = [c["hex"] for c in colors["sub"]]
    print(f"  - 메인: {main_hex} ({main_name})")
    print(f"  - 서브: {', '.join(sub_hex_list)}")

    palette_path = create_color_palette(colors, output_dir=output_dir)
    print(f"  - 저장: {palette_path}")

    print("[5/5] 로고 시안 생성 중...")
    logo_paths = generate_logos(
        brief=brief,
        colors=colors,
        client=client,
        output_dir=output_dir,
        brand_name=brand_name,
        num_images=2,
    )

    for path in logo_paths:
        print(f"  - 저장: {path}")

    return {
        "colors": colors,
        "palette_path": palette_path,
        "logo_paths": logo_paths,
    }


# -----------------------------
# 단독 실행용
# -----------------------------
if __name__ == "__main__":
    print("🎨 Visual Generator")
    brief_path = input("브리프 파일 경로를 입력하세요 (엔터 시 brief.json): ").strip() or "brief.json"
    output_dir = input("출력 폴더 경로를 입력하세요 (엔터 시 ./output): ").strip() or "./output"
    brand_name = input("브랜드명을 입력하세요 (엔터 시 푸트테크): ").strip() or "푸트테크"

    try:
        result = run_visual_pipeline(
            brief_path=brief_path,
            output_dir=output_dir,
            brand_name=brand_name,
        )
        print("\n✅ visual.py 실행 완료")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
