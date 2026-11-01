"""Official 2026 BEHAVIOR task IDs and an optional cached task-description catalog."""
from __future__ import annotations
import difflib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
TASKS_URL = 'https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/resolve/main/meta/tasks.jsonl'
TASK_NAMES = '''turning_on_radio
picking_up_trash
putting_away_Halloween_decorations
cleaning_up_plates_and_food
can_meat
setting_mousetraps
hiding_Easter_eggs
picking_up_toys
rearranging_kitchen_furniture
putting_up_Christmas_decorations_inside
set_up_a_coffee_station_in_your_kitchen
putting_dishes_away_after_cleaning
preparing_lunch_box
loading_the_car
carrying_in_groceries
bringing_in_wood
moving_boxes_to_storage
bringing_water
tidying_bedroom
outfit_a_basic_toolbox
sorting_vegetables
collecting_childrens_toys
putting_shoes_on_rack
boxing_books_up_for_storage
storing_food
clearing_food_from_table_into_fridge
assembling_gift_baskets
sorting_household_items
getting_organized_for_work
clean_up_your_desk
setting_the_fire
clean_boxing_gloves
wash_a_baseball_cap
wash_dog_toys
hanging_pictures
attach_a_camera_to_a_tripod
clean_a_patio
clean_a_trumpet
spraying_for_bugs
spraying_fruit_trees
make_microwave_popcorn
cook_cabbage
chop_an_onion
slicing_vegetables
chopping_wood
cook_hot_dogs
cook_bacon
freeze_pies
canning_food
make_pizza
freeze_fruit
cook_a_brisket
sorting_bottles_cans_and_paper
tidying_living_room
putting_away_toys
re_shelving_library_books
make_rose_centerpieces
sweeping_garage
stacking_wood
organizing_art_supplies
scrubbing_bathroom_floor
bringing_paper_to_recycling
halve_an_egg
installing_smoke_detectors
setting_the_table
unloading_the_car
turning_out_all_lights_before_sleep
boxing_food_after_dinner
cleaning_up_branches_and_twigs
vacuuming_floors
thawing_frozen_food
clean_your_rusty_garden_tools
cook_a_frozen_pie
organizing_school_stuff
carrying_out_garden_furniture
put_together_a_basic_pruning_kit
dispose_of_glass
installing_a_modem
make_cabinet_doors
polishing_shoes
clean_up_broken_glass
packing_meal_for_delivery
store_batteries
store_honey
tidying_bathroom
putting_dirty_dishes_in_sink
make_gift_bags_for_baby_showers
collecting_aluminum_cans
rearrange_your_room
installing_a_fax_machine
composting_waste
store_produce
installing_a_scanner
clean_a_keyboard
dispose_of_batteries
cook_brussels_sprouts
cook_broccolini
setup_a_bar_for_a_cocktail_party
laying_tile_floors
sorting_books_on_shelf'''.splitlines()

# Conservative, hand-verified paraphrases. Never silently turn an unsupported task into a different task.
ALIASES = {
    'モデムを設置して': 'installing_a_modem',
    'モデムの設置': 'installing_a_modem',
    'ラジオをつけて': 'turning_on_radio',
    'ラジオの電源を入れて': 'turning_on_radio',
    'リビングの空き缶をキッチンのゴミ箱に入れて': 'picking_up_trash',
    'リビングの空き缶を捨てて': 'picking_up_trash',
    '冷蔵庫の水をリビングのテーブルに持ってきて': 'bringing_water',
    'キッチンの冷蔵庫から水のボトルを2本取ってリビングのコーヒーテーブルに置いて': 'bringing_water',
    'ガレージから食料品をキッチンに運んで': 'carrying_in_groceries',
    'おもちゃを片付けて': 'putting_away_toys',
    'リビングを片付けて': 'tidying_living_room',
    '寝室を片付けて': 'tidying_bedroom',
    '汚れた皿を流しに入れて': 'putting_dirty_dishes_in_sink',
    '食器をシンクに入れて': 'putting_dirty_dishes_in_sink',
    '本を本棚に戻して': 're_shelving_library_books',
    '靴を靴棚に置いて': 'putting_shoes_on_rack',
    'テーブルをセットして': 'setting_the_table',
    '机を片付けて': 'clean_up_your_desk',
    '野菜を仕分けして': 'sorting_vegetables',
    'ガレージを掃いて': 'sweeping_garage',
    '掃除機で床を掃除して': 'vacuuming_floors',
    '電気を全部消して': 'turning_out_all_lights_before_sleep',
    '電池を片付けて': 'store_batteries',
    '玉ねぎを切って': 'chop_an_onion',
    'ブロッコリーを調理して': 'cook_broccolini',
}

class TaskSelectionError(ValueError):
    pass


def load_catalog() -> dict[str, str]:
    mapping = {name: name.replace('_', ' ') for name in TASK_NAMES}
    # Descriptions verified against the official 2026 task list; helpful offline.
    mapping.update({
        'turning_on_radio': 'リビングのテーブルにあるラジオの電源を入れる',
        'installing_a_modem': 'モデムを設置する',
        'bringing_water': 'キッチンの冷蔵庫からボトル2本を取り、リビングのコーヒーテーブルに置いて冷蔵庫を閉める',
        'picking_up_trash': 'リビングの空き缶3本をキッチンのゴミ箱へ運ぶ',
        'putting_dirty_dishes_in_sink': '汚れた皿とボウルをすべてキッチンのシンクへ運ぶ',
        'carrying_in_groceries': 'ガレージの車から食料品を取り、トマトと牛乳をキッチンの冷蔵庫へ入れる',
        'putting_shoes_on_rack': '廊下にある靴2足とサンダル2足を靴棚に並べる',
        'turning_out_all_lights_before_sleep': '寝る前にすべての照明・スイッチを消す',
        'setting_the_table': '朝食テーブルに皿、カップケーキ、ナイフとフォークを配置する',
        'putting_away_toys': '床の玩具フィギュアをすべて玩具箱へ入れる',
    })
    path = ROOT / 'data' / 'tasks.jsonl'
    if not path.is_file():
        return mapping
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        name = obj.get('task_name')
        if name in mapping and isinstance(obj.get('task'), str):
            mapping[name] = obj['task']
    return mapping


def download_catalog(url: str = TASKS_URL) -> int:
    req = urllib.request.Request(url, headers={'User-Agent': 'kadai-physicalai/1.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        text = response.read(2_000_000).decode('utf-8')
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    unique = {row.get('task_name') for row in lines}
    if not set(TASK_NAMES).issubset(unique):
        raise RuntimeError('Official task list mismatch; refusing to overwrite local cache')
    target = ROOT / 'data' / 'tasks.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(target)
    return len(lines)


def normalize(text: str) -> str:
    return re.sub(r'[\s\u3000。、！？!?.]', '', text).lower()


def select_task(instruction: str) -> tuple[str, str]:
    """Map natural language using OpenAI; explicit official task IDs need no API."""
    query = instruction.strip()
    if not query:
        raise TaskSelectionError('指示が空です')
    for task in TASK_NAMES:
        if normalize(query) == normalize(task):
            return task, '公式タスクIDを指定'
    from .openai_router import route
    task, reason = route(query, load_catalog())
    return task, 'OpenAI APIで選択' + (f'：{reason}' if reason else '')


def suggestions(query: str, limit: int = 5) -> list[str]:
    if not query:
        return TASK_NAMES[:limit]
    in_names = [task for task in TASK_NAMES if normalize(query) in normalize(task)]
    match_aliases = [v for k, v in ALIASES.items() if normalize(query) in normalize(k)]
    fuzzy = difflib.get_close_matches(query, TASK_NAMES, n=limit, cutoff=0.3)
    return list(dict.fromkeys(match_aliases + in_names + fuzzy))[:limit]
