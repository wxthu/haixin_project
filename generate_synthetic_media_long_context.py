import json
import random
import argparse
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Case:
    user_question: str
    user_intent: str  # "检索意图" or "知识意图"
    seed_titles: List[str]
    theme: str


INSTRUCTION = (
    "你是一个影视领域的信息总结专家，你的任务是根据####用户问题，####用户意图，####相关文档信息列表, ####影视剧名称列表生成准确流畅的总结内容。\n"
    "####用户问题：代表了用户的需求\n"
    "####用户意图：####用户问题的意图\n"
    "1. 检索意图，推荐一些影视剧/询问具体的影视剧，例如：“开心的时候适合看什么电影”，“真相只有一个出自哪部动漫”；\n"
    "2. 知识意图，询问影视剧、明星相关的事实信息，例如：“刘德华多少岁了”，“《摔跤吧爸爸》的豆瓣评分是多少”。\n"
    "####相关文档信息列表, 包含影视推荐理由、事实数据（如评分、演员信息）等内容的素材。\n"
    "####影视剧名称列表：从####相关文档信息列表中抽取的影视剧的名称列表，可能为空。\n\n"
    "你需要按照以下步骤一步一步的进行思考然后进行总结:\n"
    "1. ####用户意图是检索意图，总结内容字数不超过68个字。\n"
    "  a. 如果是推荐一些影视剧，总结特点，但总结里不要出现任何一个影视名称。\n"
    "  b. 如果是询问具体影视，若列表中能找到对应影视，则名称用《》括起来；若找不到，先如实回答，再委婉推荐其他作品。\n"
    "  c. 如果素材里没有符合问题的内容，根据素材生成总结，说明推荐依据与亮点，语言尽量亲切、口语化。\n"
    "2. ####用户意图是知识意图，总结内容字数不超过250个字。\n"
    "  a. 若素材相关，精准提取事实并口语化回答，再用一句话概述推荐作品与问题的关系。\n"
    "  b. 若素材不相关，不要捏造事实，可采用通用性回复。\n"
    "  c. 不要出现敏感词汇；只输出中文；不要包裹引号；需要 markdown 时严格遵守格式。\n"
    "  d. 询问上映时间而素材无年份时，默认当前年份为2025年。\n"
    "  e. 询问演员是否参演等问题，需要通读素材并判断。\n"
)


SAFE_TITLES = [
    "海边的曼彻斯特",
    "寻梦环游记",
    "千与千寻",
    "龙猫",
    "怦然心动",
    "飞屋环游记",
    "当幸福来敲门",
    "放牛班的春天",
    "哈利·波特与魔法石",
    "小森林",
    "少年派的奇幻漂流",
    "盗梦空间",
    "星际穿越",
    "美丽人生",
    "海蒂和爷爷",
]


CASES: List[Case] = [
    Case("适合周末放松的温暖电影推荐", "检索意图", ["怦然心动", "飞屋环游记", "小森林"], "治愈、温暖、轻松"),
    Case("《寻梦环游记》讲了什么故事？", "知识意图", ["寻梦环游记"], "剧情梗概、主题与亮点"),
    Case("《千与千寻》的导演是谁？", "知识意图", ["千与千寻"], "主创信息与获奖口碑"),
    Case("想看亲子主题、适合全家一起看的片子，有推荐吗", "检索意图", ["龙猫", "海蒂和爷爷", "寻梦环游记"], "亲子、家庭、成长"),
    Case("《放牛班的春天》主要演员有哪些？", "知识意图", ["放牛班的春天"], "演员信息与看点"),
    Case("有没有节奏不快但很有力量的励志片", "检索意图", ["当幸福来敲门", "美丽人生"], "励志、人生、希望"),
    Case("《少年派的奇幻漂流》获得过哪些重要奖项？", "知识意图", ["少年派的奇幻漂流"], "奖项与口碑事实"),
    Case("想看奇幻冒险、适合入门的系列电影", "检索意图", ["哈利·波特与魔法石"], "奇幻、冒险、成长线"),
    Case("《星际穿越》里关于时间的设定是什么？", "知识意图", ["星际穿越"], "设定解释与观影提示"),
    Case("《小森林》适合什么心情的时候看？", "检索意图", ["小森林"], "氛围、节奏、适配情绪"),
    Case("适合通勤路上刷的轻松短片风格电影推荐", "检索意图", ["飞屋环游记", "龙猫"], "轻松、明快、治愈"),
    Case("《美丽人生》表达的核心主题是什么？", "知识意图", ["美丽人生"], "主题解读与观影建议"),
    Case("有没有适合失眠夜晚看的安静电影", "检索意图", ["小森林", "海边的曼彻斯特"], "安静、细腻、慢节奏"),
    Case("《盗梦空间》的主要人物关系怎么理解？", "知识意图", ["盗梦空间"], "人物关系与叙事结构"),
    Case("适合和朋友一起看的高口碑动画电影推荐", "检索意图", ["千与千寻", "寻梦环游记", "龙猫"], "动画、口碑、合家欢"),
    Case("《当幸福来敲门》结局是什么？", "知识意图", ["当幸福来敲门"], "剧情要点与情绪落点"),
    Case("想找一些带点冒险但不吓人的电影", "检索意图", ["哈利·波特与魔法石", "少年派的奇幻漂流"], "冒险、奇幻、轻松"),
    Case("《海蒂和爷爷》适合多大孩子看？", "知识意图", ["海蒂和爷爷"], "适龄与观影提示"),
    Case("适合雨天看的温柔系电影，有吗", "检索意图", ["怦然心动", "小森林"], "雨天、温柔、治愈"),
    Case("《飞屋环游记》为什么那么打动人？", "知识意图", ["飞屋环游记"], "情感点与叙事亮点"),
]


def _mk_doc_block(case: Case, idx: int, extra_len_hint: int) -> List[str]:
    rng = random.Random(idx * 10007 + extra_len_hint)
    titles = case.seed_titles[:] or [rng.choice(SAFE_TITLES)]
    related = []

    # Mix of "网页标题" and "媒资" style snippets.
    for k in range(1, 28):
        t = rng.choice(titles + SAFE_TITLES)
        related.append(
            f"网页标题: 观影笔记|《{t}》适合{case.theme}的理由（第{k}篇） "
            f"网页内容: 文章从氛围、节奏、人物关系与主题表达四个角度梳理《{t}》，"
            f"并给出不剧透的看点提示，如镜头语言、配乐风格、情绪曲线与适合的观影场景。"
        )
        related.append(
            f"媒资名称:{t}, 类型:电影, 关键词:{case.theme}, 简介:围绕“{case.theme}”展开，"
            f"兼具故事性与情绪感染力；适合在安静的时间段沉浸式观看。"
        )
        # Add some fact-like lines (still synthetic).
        related.append(
            f"网页标题: 《{t}》口碑与亮点盘点（第{k}次更新） "
            f"网页内容: 整体评价集中在“情感真挚”“节奏舒缓”“细节耐品”等方向；"
            f"若你喜欢{case.theme}相关作品，可以把它加入片单。"
        )

    # Add a few long, multi-sentence entries to push length upward.
    for k in range(1, 8):
        t = rng.choice(titles + SAFE_TITLES)
        related.append(
            f"网页标题: 深度解析《{t}》的叙事结构与情绪铺陈（长文{k}） "
            f"网页内容: 该文分为“开端的情绪基调”“中段的冲突与转折”“结尾的余味”三部分，"
            f"同时总结了观众常提到的亮点：人物成长的层次、关系变化的自然、以及关键场景中"
            f"通过光影与环境音营造的沉浸感。文末还给出同类型作品清单，便于延展观看。"
        )

    rng.shuffle(related)
    return related


def _mk_titles_list(case: Case) -> str:
    if not case.seed_titles:
        return "[]"
    # Keep original dataset style: sometimes with 《》, sometimes plain; we standardize with 《》.
    return "[" + ",".join([f"《{t}》" for t in case.seed_titles]) + "]"


def make_item(case: Case, idx: int, min_len: int, max_len: int) -> dict:
    docs = _mk_doc_block(case, idx, min_len + max_len)
    titles_list = _mk_titles_list(case)

    # Assemble content with explicit markers as in the dataset.
    content_parts = [
        INSTRUCTION.rstrip("\n"),
        f"\n####用户问题：{case.user_question}\n"
        f"####用户意图：{case.user_intent}\n"
        "####相关文档信息列表:[" + "\n".join(docs) + "]\n"
        f"####影视剧名称列表：{titles_list}\n"
        "####总结内容：",
    ]
    content = "\n".join(content_parts)

    # If too short, pad by repeating additional synthetic doc lines.
    pad_rng = random.Random(900000 + idx)
    while len(content) < min_len:
        t = pad_rng.choice((case.seed_titles or SAFE_TITLES))
        extra = (
            f"\n网页标题: 片单补充|与《{t}》气质相近的延展推荐 "
            f"网页内容: 从“{case.theme}”出发，补充若干风格相近的作品方向，"
            f"例如更偏生活流的叙事、更偏童话感的想象力、或更偏细腻情感的表达。"
        )
        insert_at = content.rfind("]\n####影视剧名称列表：")
        if insert_at == -1:
            content += extra
        else:
            content = content[:insert_at] + extra + content[insert_at:]

    # If too long, truncate the docs portion (keep structure intact).
    if len(content) > max_len:
        # Keep head and tail, shorten doc list by slicing characters inside the list.
        head = INSTRUCTION.rstrip("\n") + f"\n####用户问题：{case.user_question}\n####用户意图：{case.user_intent}\n####相关文档信息列表:["
        tail = f"]\n####影视剧名称列表：{titles_list}\n####总结内容："
        budget = max_len - len(head) - len(tail)
        if budget < 0:
            # Fallback: extremely small budget; just hard cut (should not happen with current params)
            content = (head + tail)[:max_len]
        else:
            doc_blob = "\n".join(docs)
            doc_blob = doc_blob[:budget]
            # avoid cutting mid surrogate / keep it clean-ish
            content = head + doc_blob + tail

    return {"content": content}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="synthetic_media_long_context_10.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-idx", type=int, default=1)
    parser.add_argument("--min-len", type=int, default=10_000)
    parser.add_argument("--max-len", type=int, default=20_000)
    args = parser.parse_args()

    out_path = args.out
    min_len, max_len = args.min_len, args.max_len

    chosen_cases: List[Case] = []
    for i in range(args.count):
        chosen_cases.append(CASES[(args.start_idx - 1 + i) % len(CASES)])

    items = []
    for j, case in enumerate(chosen_cases, start=0):
        item_idx = args.start_idx + j
        item = make_item(case, item_idx, min_len, max_len)
        if not (min_len <= len(item["content"]) <= max_len):
            raise RuntimeError(f"item idx={item_idx} length out of range: {len(item['content'])}")
        items.append(item)

    mode = "a" if args.append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(out_path)
    for i, item in enumerate(items, start=args.start_idx):
        print(i, len(item["content"]))


if __name__ == "__main__":
    main()

