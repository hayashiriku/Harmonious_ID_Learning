"""
weather_module.py
=================
OpenWeatherMap One Call API 3.0 を使った気象情報取得モジュール。
リアルタイム（現在地・現在時刻）と研究用（過去データ）の両方に対応。

使い方:
    # リアルタイム
    wm = WeatherModule(api_key="YOUR_API_KEY")
    result = wm.get(lat=35.6895, lon=139.6917)

    # 研究用（過去データ）
    result = wm.get(lat=35.6895, lon=139.6917, dt=1718000000)

    print(result)
    # {
    #   "weather":     "sunny",
    #   "time_of_day": "evening",
    #   "season":      "summer",
    #   "raw":         { ... }   ← APIの生レスポンス（デバッグ用）
    # }
"""

import requests
import datetime
from typing import Optional

# ──────────────────────────────────────────────
# 定数：weather_id → ラベル変換テーブル
# ──────────────────────────────────────────────
WEATHER_MAP = [
    (range(200, 300), "rainy"),    # 雷雨     → rainy
    (range(300, 400), "rainy"),    # 霧雨     → rainy
    (range(500, 600), "rainy"),    # 雨       → rainy
    (range(600, 700), "snowy"),    # 雪
    (range(700, 800), "cloudy"),   # 霧・靄   → cloudy
    ([800],           "sunny"),    # 快晴
    (range(801, 805), "cloudy"),   # 曇り
]

# 月 → 季節（北半球）
SEASON_MAP = {
    1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
    12: "winter",
}


class WeatherModule:
    """
    OpenWeatherMap One Call API 3.0 ラッパー。
    リアルタイムと過去データを同一インターフェースで取得する。
    """

    BASE_REALTIME   = "https://api.openweathermap.org/data/3.0/onecall"
    BASE_HISTORICAL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

    def __init__(self, api_key: str):
        """
        Parameters
        ----------
        api_key : str
            OpenWeatherMap の API キー。
            One Call by Call プランへの登録が必要（1,000回/日 無料）。
        """
        self.api_key = api_key

    # ──────────────────────────────────────────────
    # 公開メソッド
    # ──────────────────────────────────────────────
    def get(
        self,
        lat: float,
        lon: float,
        dt:  Optional[int] = None,
    ) -> dict:
        """
        気象情報を取得してラベルに変換して返す。

        Parameters
        ----------
        lat : float   緯度
        lon : float   経度
        dt  : int, optional
            UNIXタイムスタンプ（UTC）。
            指定なし → リアルタイム取得。
            指定あり → 過去データ取得（1979年1月1日以降）。

        Returns
        -------
        dict
            weather     : str   天気ラベル（"sunny" / "rainy" など）
            time_of_day : str   時間帯（"dawn" / "daytime" / "evening" / "night"）
            season      : str   季節（"spring" / "summer" / "autumn" / "winter"）
            raw         : dict  APIの生レスポンス（デバッグ用）
        """
        raw = self._fetch(lat, lon, dt)
        return self._parse(raw)

    # ──────────────────────────────────────────────
    # 内部メソッド：API呼び出し
    # ──────────────────────────────────────────────
    def _fetch(self, lat: float, lon: float, dt: Optional[int]) -> dict:
        """APIを呼び出して生レスポンスの data フィールドを返す。"""

        if dt is None:
            # ── リアルタイム ──
            params = {
                "lat":     lat,
                "lon":     lon,
                "appid":   self.api_key,
                "exclude": "minutely,hourly,daily,alerts",
            }
            resp = requests.get(self.BASE_REALTIME, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()["current"]

        else:
            # ── 過去データ（研究用） ──
            params = {
                "lat":   lat,
                "lon":   lon,
                "dt":    dt,
                "appid": self.api_key,
            }
            resp = requests.get(self.BASE_HISTORICAL, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()["data"][0]  # 過去データは data リストの先頭

    # ──────────────────────────────────────────────
    # 内部メソッド：ラベル変換
    # ──────────────────────────────────────────────
    def _parse(self, data: dict) -> dict:
        """生レスポンスの data フィールドを受け取り、ラベル辞書を返す。"""

        weather_id = data["weather"][0]["id"]
        dt_unix    = data["dt"]
        sunrise    = data.get("sunrise", None)
        sunset     = data.get("sunset",  None)

        return {
            "weather":     self._get_weather_label(weather_id),
            "time_of_day": self._get_time_of_day(dt_unix, sunrise, sunset),
            "season":      self._get_season(dt_unix),
            "raw":         data,
        }

    # ──────────────────────────────────────────────
    # 変換ヘルパー
    # ──────────────────────────────────────────────
    @staticmethod
    def _get_weather_label(weather_id: int) -> str:
        """weather_id（OpenWeatherMap 気象コード）→ 天気ラベル"""
        for id_range, label in WEATHER_MAP:
            if weather_id in id_range:
                return label
        return "unknown"

    @staticmethod
    def _get_time_of_day(
        dt:      int,
        sunrise: Optional[int],
        sunset:  Optional[int],
    ) -> str:
        """
        UNIXタイムと日の出/日の入り時刻から時間帯ラベルを返す。

        dawn    : 日の出 ± 1時間
        daytime : 日の出+1h 〜 日の入り-1h
        evening : 日の入り-1h 〜 日の入り+1.5h  （ゴールデンアワー含む）
        night   : それ以外
        """
        if sunrise is None or sunset is None:
            # 日の出/日の入りが取れない（極地など）→ UTC時刻から簡易判定
            hour = datetime.datetime.fromtimestamp(
                dt, tz=datetime.timezone.utc
            ).hour
            if   6  <= hour < 9:  return "dawn"
            elif 9  <= hour < 17: return "daytime"
            elif 17 <= hour < 20: return "evening"
            else:                 return "night"

        if   dt < sunrise + 3600:  return "dawn"
        elif dt < sunset  - 3600:  return "daytime"
        elif dt < sunset  + 5400:  return "evening"  # +1.5h
        else:                      return "night"

    @staticmethod
    def _get_season(dt: int) -> str:
        """UNIXタイムスタンプ（UTC）→ 季節ラベル（北半球）"""
        month = datetime.datetime.fromtimestamp(
            dt, tz=datetime.timezone.utc
        ).month
        return SEASON_MAP[month]


# ──────────────────────────────────────────────
# テキスト変換ユーティリティ
# ──────────────────────────────────────────────
def build_scene_text(weather_result: dict, scene_type: str = "") -> str:
    """
    WeatherModule.get() の結果と情景タイプを受け取り、
    Text2Tracks に渡すシーンテキストを組み立てる。

    Parameters
    ----------
    weather_result : dict   WeatherModule.get() の戻り値
    scene_type     : str    画像処理で取得した情景タイプ（例: "seaside"）

    Returns
    -------
    str  例: "sunny summer evening seaside"
    """
    parts = [
        weather_result["weather"],
        weather_result["season"],
        weather_result["time_of_day"],
    ]
    if scene_type:
        parts.append(scene_type)

    return " ".join(parts)


# ──────────────────────────────────────────────
# 動作確認（APIキーなしでモックテスト）
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # モック：APIを呼ばずに変換ロジックをテスト
    mock_realtime = {
        "dt":      1718100000,   # 2024-06-11 15:00 UTC
        "sunrise": 1718058000,   # 05:00 UTC
        "sunset":  1718110800,   # 19:00 UTC
        "weather": [{"id": 800, "main": "Clear", "description": "clear sky"}],
    }

    mock_historical = {
        "dt":      1696000000,   # 2023-09-29 12:00 UTC
        "sunrise": 1695958800,
        "sunset":  1696001400,
        "weather": [{"id": 803, "main": "Clouds", "description": "broken clouds"}],
    }

    wm = WeatherModule.__new__(WeatherModule)  # APIキー不要でインスタンス化

    print("=== リアルタイム（モック） ===")
    r1 = wm._parse(mock_realtime)
    print(r1)
    print("scene_text:", build_scene_text(r1, scene_type="seaside"))

    print("\n=== 過去データ（モック） ===")
    r2 = wm._parse(mock_historical)
    print(r2)
    print("scene_text:", build_scene_text(r2, scene_type="mountain road"))