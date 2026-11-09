# Text Speak API 2.2

OmniVoice server API: tayyor ovozlar, tavsifdan ovoz yaratish, avtomatik ovoz,
audio namunadan bir martalik klonlash, ovoz profilini saqlash, audioni matnga
o'girish, ko'p matnni navbatga qo'yish va batch'dan ZIP audio olish.
API tarjima yoki chat javoblarini yaratmaydi.

## GitHub'dan yuklab ishga tushirish

Python 3.11, Git va internet kerak. Tavsiya etilgan muhit: Linux yoki Windows'da
WSL2. Quyidagi yo'l public PyTorch backendidan foydalanadi; repodagi asosiy
`Dockerfile`/`compose.yaml` esa oldindan tayyorlangan server image'i, offline
paketlar va TensorRT volume'iga mo'ljallangan.

```bash
git clone https://github.com/mirazizGG/text-speak.git
cd text-speak
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# NVIDIA GPU uchun (mos drayver talab qilinadi):
python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.local.txt
cp .env.example .env
```

CPU ishlatish uchun Torch o'rnatish qatorida indeksni
`https://download.pytorch.org/whl/cpu` ga almashtiring va `.env` ichida
`TTS_DEVICE=cpu` yozing. CPU tezligi bu o'zgarishda sinovdan o'tkazilmagan.
NVIDIA rejimida asosiy modeldan tashqari alohida speech worker ham xotira oladi;
startup GPU/RAM zaxirasini tekshiradi. 8 GB GPU yetishi kafolatlanmaydi.

`.env` dagi `TTS_API_KEY` o'rniga o'zingizning kamida 32 belgili tasodifiy kalitingizni
yozing; yangi kalitni `python -c "import secrets; print(secrets.token_hex(32))"`
bilan yarating. `.env` shaxsiy bo'lib qoladi. Tayyor bo'lgach loyiha papkasidan:

```bash
python -m uvicorn api:create_app --factory --env-file .env --host 127.0.0.1 --port 8070 --workers 1
```

**Dastur modelni o'zi tayyorlaydi.** Ishga tushishda mahalliy model fayllari
tekshiriladi; yetishmagan yoki kesilgan fayllar rasmiy Hugging Face manbasidan
yuklanadi. Yangi yuklamaning SHA256 xeshi tekshiriladi, so'ng tayyor fayl o'rniga
qo'yiladi. Keyingi ishga tushishda mavjud katta vaznlar hajmi, kichik
metadata/tokenizer fayllari esa xeshi bilan tekshiriladi; to'liq nusxa uchun
tarmoq yoki qayta yuklash kerak emas. Jarayon/progress terminalda ko'rinadi.

| Vazifa | Model va aniq versiya | Mahalliy papka | Hajm |
|---|---|---|---|
| Matndan ovoz, audio tokenizer bilan | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice/tree/c5fdb5ccb189668d56333f77ba2629f4cd7535f4), `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` | `offline/model` | taxminan 3.27 GB |
| Transkripsiya va klonlash uchun avtomatik matn | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9), `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | `offline/whisper` | taxminan 1.62 GB |

Birinchi ishga tushish taxminan 4.9 GB model yuklaydi va tayyor preset ovozlarini
yaratadi; paketlar/keshlar/natijalar uchun qo'shimcha disk va RAM kerak.
Yuklama uzilsa dastur xatoni ko'rsatadi; internet/joy muammosini tuzatib qayta
ishga tushiring. To'liq fayllar saqlanadi, tugallanmagan fayl tayyor model sifatida
ishlatilmaydi. Model papkasi yozishga ruxsatli bo'lishi kerak.

`http://127.0.0.1:8070/readyz` 200 qaytargach,
`http://127.0.0.1:8070/docs` ni oching va **Authorize** orqali o'z API kalitingizni
kiriting. Model yuklanayotganida/generatsiya boshlanguncha tayyorlik 503 bo'lishi mumkin.

Internetli avtomatik tayyorlashni o'chirish yoki o'zingiz tayyorlagan boshqa
mahalliy ASR snapshotidan foydalanish uchun `TTS_AUTO_DOWNLOAD_MODELS=0`.
`HF_HUB_OFFLINE=1` yoki `TRANSFORMERS_OFFLINE=1` ham tarmoqqa chiqishni taqiqlaydi;
bu rejimda modellarni oldindan tayyorlash kerak. Server Docker image'ida offline
rejim allaqachon yoqilgan. TensorRT/FlashInfer tezlashtirishlari alohida, mos
GPU/runtime talab qiladi va boshqa kompyuterga tayyor plan ko'chirish shart emas.
Model litsenziyalari: [OmniVoice model kartasi](https://huggingface.co/k2-fsa/OmniVoice)
va [Whisper model kartasi](https://huggingface.co/openai/whisper-large-v3-turbo).

Ovoz sifati uchun standart generatsiya 32 qadamda qoladi. Speech ishchilari
FlashInfer'ning asl CUDA graph yo'lidan foydalanishi mumkin: shakl bir xil bo'lsa
graf qayta ishlatiladi, xotira yoki shakl mos kelmasa odatiy FlashInfer ishlaydi.
Har ishchining graf keshi son va taxminiy xotira bo'yicha chegaralangan; bu matn
soni yoki uzunligiga kvota emas. `TTS_FLASHINFER_GRAPHS=0` grafni o'chiradi.
Profil yaratish va transkripsiya modeli odatiy rejimda qoladi.

Ko'p qatorli matnda so'zlar qo'shilib ketmasligi uchun qator uzilishlari bo'shliqqa
aylanadi. Yangi ovoz saqlashda yozuv tili ko'rsatilsa va mahalliy Whisper uni
qo'llasa, avtomatik transkripsiya shu tilda bajariladi. Tayyor saqlangan profillar
qayta yozilmaydi; qo'lda berilgan to'g'ri transkript ustun turadi.

## Ovoz studiyasi

`/docs` sahifasining yuqorisidagi **Ovoz studiyasi** dan bir oynada foydalaning:
**Authorize orqali kirish** tugmasida API kalitini kiriting. Avval saqlangan ovozni
ro'yxatdan tanlang yoki yangi audio fayl, ovoz nomi va turini kiritib **Ovozni saqlash**
ni bosing. Saqlangach, **Matn yozish** oynasi ochiladi; matn va ixtiyoriy fayl nomini
kiriting, keyin **Ovozni yaratish** ni bosing. Tayyor audio yonida **Yuklab olish**
tugmasi paydo bo'ladi; audioni tinglab, xohlasangiz alohida yuklab oling. Saqlangan ovozni tanlab, tasdiqlash orqali
o'chirish ham mumkin. API hujjati va endpointlar sahifaning pastki qismida qoladi.

**Ko'p matn** usulida 50/100 yoki undan ko'p matnni bo'sh qator bilan ajrating,
yoki `[{"text":"Salom","key":"salom"}]` JSON ro'yxatini kiriting; TXT/JSON faylni
ham ochish mumkin. Matn soni va uzunligiga standart dasturiy kvota qo'yilmaydi.
Tanlangan ovoz va til barcha matnlarga qo'llanadi. Fayl nomi prefiksi `lesson`
bo'lsa, nomlar `lesson.001.wav`, `lesson.002.wav`; bo'sh bo'lsa `audio.001.wav`
dan boshlanadi. JSON dagi alohida `key` shu element nomini belgilaydi.
Yuborish qabul qilingach, keyingi matnlarni darhol qo'shish mumkin. Har bir matnning
holati, tinglash va yuklash tugmasi alohida; ko'p fayl avtomatik yuklanmaydi.
Bitta matn uchun ham fayl faqat **Yuklab olish** bosilganda yuklanadi. Qayta kirganda saqlangan oxirgi
so'rovlar qayta ko'rinadi; ovoz tanlovini o'zgartirish oldingi natijalarni o'zgartirmaydi.

## Ulanish va Postman

- Base URL: `http://10.2.1.150:8070`; interaktiv hujjat: `/docs`; OpenAPI: `/openapi.json`.
- Kompyuter serverga lokal tarmoq yoki VPN orqali kira olishi kerak.
- Barcha `/v1/` so'rovlariga `Authorization: Bearer YOUR_API_KEY` yuboriladi.
- Kalit `API-ACCESS.env` ichida. Postman collection va `Text-Speak-API.txt` ham kalitni o'z ichiga oladi; paketni faqat foydalanadigan dasturchiga bering.
- Postman: **Import -> Files -> Text-Speak.postman_collection.json**. Collection darajasida Bearer kaliti tayyor.
- Collection Variables: `base_url`, `voice_id` (boshlang'ich qiymati `female_1`). Profil yaratish 201 qaytarsa, skript `voice_id` ni yangi `id` bilan yangilaydi.
- Audio natijasi uchun **Send and Download** bilan `speech.wav`, batch uchun `audio-batch.zip` saqlang. Audio javobi JSON emas.
- Upload so'rovlarida **Body -> form-data -> Select Files** orqali audio namunangizni tanlang. Multipart `Content-Type` headerini qo'lda yozmang; boundary'ni Postman o'zi qo'yadi.
- `ref_text` maydoni dastlab o'chirilgan: Whisper transkriptni o'zi oladi. Aniq transkript berish uchun maydon yonidagi belgini yoqing.

Kalitni ommaviy brauzer JavaScript kodiga joylamang; boshqa dastur o'z backendidan murojaat qilsin.

## Endpointlar

Barcha `/v1/` endpointlar Bearer kalitini talab qiladi; holat tekshiruvlari ochiq.

| Usul | Yo'l | Kirish va natija |
|---|---|---|
| GET | `/healthz` | Ochiq. API jarayoni ishlasa 200. |
| GET | `/readyz` | Ochiq. Model/ovozlar tayyor bo'lsa 200, aks holda 503. |
| GET | `/v1/languages` | `data`: tillarning `code`, `name` qiymatlari. |
| GET | `/v1/voices` | `data`: tayyor va saqlangan ovozlar. `?gender=female` yoki `?gender=male` bilan filtrlash. |
| GET | `/v1/capabilities` | ASR/normalizatsiya mavjudligi, speaker atributlari, generatsiya sxemasi va limitlar. |
| GET | `/v1/voices/{voice_id}/sample` | Tanlangan ovoz namunasi, WAV; klon namunasida asl sample rate saqlanadi. Noma'lum `id`: 404. |
| POST | `/v1/audio/speech` | JSON: preset/design/auto rejimida matndan WAV. |
| POST | `/v1/audio/jobs` | JSON: speech elementlari ro'yxati; 202 va saqlangan guruh/ish holatlari. |
| GET | `/v1/audio/jobs` | Oxirgi 20 ta guruh (`data`) va faol parallelizm/navbat hisoblari (`execution`). |
| GET | `/v1/audio/jobs/{group_id}` | Bitta guruh, uning `jobs` elementlari va `counts`. |
| GET | `/v1/audio/jobs/{group_id}/{job_id}/audio` | Faqat tayyor ishning WAV fayli; `Content-Disposition` da fayl nomi. |
| POST | `/v1/voices/design` | JSON: tavsifdan saqlanadigan ovoz. 201 va profil metama'lumoti. |
| POST | `/v1/voices/clone` | Multipart: namunadan saqlanadigan ovoz. 201 va profil metama'lumoti. |
| DELETE | `/v1/voices/{voice_id}` | Saqlangan klon/design ovozini o'chirish. 200 va `{"id":"...","deleted":true}`. Tayyor presetlar himoyalangan. |
| POST | `/v1/audio/clone` | Multipart: audio namunadan bir martalik ovoz, WAV. |
| POST | `/v1/audio/transcriptions` | Multipart: qisqa audio namunasidan `{"text":"..."}`. |
| POST | `/v1/audio/batch` | JSON: matnlar, WAV fayllari va `manifest.json` bo'lgan ZIP. |

Capabilities javobidagi asosiy kalitlar: `asr_available`, `asr_loaded`,
`normalization_languages`, `normalization_modes`, `generation_fields`,
`generation_schema`, `voice_design_attributes`, `modes`, `limits`, `reference_audio_formats`, `nonverbal_tags`,
`pronunciation`, `voice_design_training_languages`, `max_custom_voices`, `custom_quota_bytes`.
`modes` ichidagi `clone` alohida `/v1/audio/clone` endpointidir; speech JSON ichidagi
`mode` qiymatlari faqat `preset`, `design`, `auto`.

## 50/100 va undan ko'p matnni navbatga qo'yish

`POST /v1/audio/jobs` matnlarni diskdagi SQLite navbatiga saqlab, audio tayyor
bo'lishini kutmasdan **202** qaytaradi. Masalan, shu kichik so'rovdagi `items`
ro'yxatini 50/100 yoki undan ko'p element bilan yuborish mumkin:

```json
{"items":[
  {"text":"Assalomu alaykum.","language":"uz","voice":"female_1","key":"lesson.001"},
  {"text":"Bugungi mashg'ulotni boshlaymiz.","language":"uz","voice":"female_1","key":"lesson.002"}
]}
```

Har element odatiy speech maydonlarini qabul qiladi; standart sifat
`generation.num_step=32` bo'lib qoladi. Javobdagi guruh `id`, `state`, `jobs` va
`counts` ni saqlang. Har bir ishda `id`, `state`, `text`, `voice_name`, `filename`,
vaqtlar, `error` va tayyor bo'lsa `download_url` bor. Holatlar: `queued`, `running`,
`succeeded`, `failed`; bir element xatosi qolganlarini bekor qilmaydi.

`GET /v1/audio/jobs/{group_id}` ni taxminan soniyada bir marta tekshiring va
`succeeded` ishning `download_url` manzilidan WAV ni oling. Holat va yuklash
so'rovlariga ham o'sha `Authorization: Bearer ...` sarlavhasi kerak. Namuna
nomlari `lesson.001.wav` va `lesson.002.wav`; `key` berilmasa `speech.wav`.
`GET /v1/audio/jobs` dagi `execution.parallel_workers` haqiqiy faol o'rinlar
sonini, `queued` va `running` esa navbat holatini ko'rsatadi.

Bo'shagan ish o'rniga navbatdagi matn darhol olinadi. Qabul qilingan ishlar
brauzer yopilsa, mijoz ulanishi uzilsa yoki API qayta ishga tushsa saqlanadi;
uzilgan ishlar qayta tiklanadi. Tayyor fayllar va yakunlangan ishlar standart
avtomatik muddatsiz saqlanadi (`TTS_JOB_RETENTION_SECONDS`). Kutayotgan ishlar shu muddat
sababli o'chirilmaydi. Ovoz namunasi qabul vaqtida alohida saqlanadi: keyin
ommaviy ovoz profili o'chirilsa ham qabul qilingan matnlar davom etadi. Oxirgi
kutayotgan/ishlayotgan ish tugagach, bu xususiy nusxa avtomatik tozalanadi.

Eski `POST /v1/audio/speech` ham shu navbatga kiradi va WAV tayyor bo'lishini
kutadi. Navbatda kutish muddati tugamaydi; `TTS_REQUEST_TIMEOUT=180` ish worker'da
boshlanganidan keyingi bajarilish vaqtiga taalluqli. Uzun navbatda mijoz yoki
proxy HTTP timeoutiga bog'lanib qolmaslik uchun yangi jobs endpointidan foydalaning.

## Matnni ovozga aylantirish

Avvalgi so'rovlar ishlashda davom etadi; `mode` yozilmasa `preset` ishlatiladi:

```json
{
  "key": "assistant.welcome",
  "text": "Assalomu alaykum. Bugun ob-havo juda yaxshi.",
  "language": "uz",
  "gender": "female",
  "voice": "female_1"
}
```

- `text`: majburiy, bo'sh bo'lmagan matn; matn tanlangan tilda yoziladi.
- `key`: ixtiyoriy fayl nomi kaliti. `"assistant.welcome"` yuborilsa, yuklash nomi `assistant.welcome.wav` bo'ladi; berilmasa yoki `null` bo'lsa, `speech.wav`. Kalit 1-128 ta ASCII harf, raqam, `_`, `-` va nuqta bilan ajratilgan bo'sh bo'lmagan qismlardan iborat; papka yo'llari va Windows rezerv nomlari qabul qilinmaydi. Bu autentifikatsiya kalitidan alohida maydon.
- `language`: ixtiyoriy, `/v1/languages` dagi kod (`uz`, `ru`, `en`) yoki `null`. `null` modelga aniq til belgisini bermaydi, tarjima qilmaydi.
- `preset`: `voice` majburiy. `gender` ixtiyoriy va profil jinsiga mos bo'lishi shart. `instruct` berilsa profil bilan mos speaker atributlari bo'lishi kerak.
- `design`: `instruct` majburiy, `voice` yuborilmaydi. `gender` ixtiyoriy va tavsifga mos bo'lishi kerak. Bu rejim profilni saqlamaydi.
- `auto`: `voice`, `instruct`, `gender` yuborilmaydi; model ovozni o'zi tanlaydi.
- Presetlar: `female_1` (Dilnoza), `female_2` (Madina), `male_1` (Jasur), `male_2` (Sardor). Bular sun'iy profillar, boshlang'ich tili o'zbekcha.
- Saqlangan custom profilning `id` qiymati keyingi preset so'rovida `voice` bo'ladi.
- Boshqa tilda klonlashda namuna tilining aksenti o'tishi mumkin; barcha tillardagi talaffuz sifati bir xil emas.

Bir martalik tavsifdan ovoz:

```json
{"text":"Hello. This voice is designed.","language":"en","mode":"design",
 "instruct":"female, young adult, moderate pitch","seed":123}
```

Avtomatik ovoz:

```json
{"text":"Assalomu alaykum.","language":"uz","mode":"auto"}
```

Natija: `Content-Type: audio/wav`, 24000 Hz, mono, PCM16. Javobni fayl yoki bytes sifatida saqlang.

Frontend yuboradigan namuna:

```json
{"key":"assistant.welcome","text":"Davom etish uchun tilni tanlang.","language":"uz","voice":"female_1"}
```

Javob sarlavhasi: `Content-Disposition: attachment; filename="assistant.welcome.wav"`.
Frontend Blob orqali yuklasa, `download` atributida shu nomni ishlatsin:

```javascript
const filename = response.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1] ?? "speech.wav";
downloadLink.download = filename;
```

Sozlangan CORS originlari uchun bu sarlavha ochilgan. Backend proxy ishlatilsa,
u ham sarlavhani frontendga uzatishi kerak. Frontenddagi `download="speech.wav"`,
curl'dagi `--output speech.wav` yoki Python'dagi `Path("speech.wav")` server nomini almashtiradi.

## Tezlik, davomiylik va barcha generatsiya sozlamalari

Speech, jobs va batch elementlari bu maydonlarni qabul qiladi. Bir martalik klonlashda
ular `options` form maydonidagi JSON orqali beriladi.

| Maydon | Standart | Chegara / vazifa |
|---|---|---|
| `speed` | `null` (1x) | 0.5-2.0; kattaroq qiymat tezroq o'qiydi. |
| `duration` | `null` | 1-30 soniya; `speed` o'rniga ishlaydi va uzun matnni chunklash yo'lini chetlab o'tadi. |
| `seed` | `null` | Butun son, 0-4294967295. |
| `normalize_text` | `false` | Son/sana/valyutani o'qishga tayyorlash; til ro'yxatini capabilities'dan tekshiring. |
| `generation.num_step` | 32 | Butun son, 1-64. 16 tezroq; kam qadam sifatga ta'sir qilishi mumkin. |
| `generation.guidance_scale` | 2.0 | 0-10. |
| `generation.t_shift` | 0.1 | 0 dan katta, 1 gacha. |
| `generation.position_temperature` | 5.0 | 0-10. |
| `generation.class_temperature` | 0.0 | 0-10. |
| `generation.layer_penalty_factor` | 5.0 | 0-10. |
| `generation.denoise` | `true` | Namuna shovqinini ko'chirmaslik boshqaruvi. |
| `generation.preprocess_prompt` | `true` | Reference audio/matnini oldindan tayyorlash. |
| `generation.postprocess_output` | `true` | Natijadagi uzun jimliklarni qisqartirish. |
| `generation.audio_chunk_duration` | 15.0 | 3-30 soniya, uzun matn bo'laklari uchun nishon. |
| `generation.audio_chunk_threshold` | 30.0 | 3-60 soniya; chunk duration'dan kichik bo'lmaydi. |
| `generation.pad_duration` | 0.1 | 0-1 soniya; chetlarga jimlik qo'shish. |
| `generation.fade_duration` | 0.1 | 0-1 soniya; audio boshi/oxirini yumshatish. |

Noma'lum maydonlar, noto'g'ri turlar, NaN/Infinity qiymatlar 422 qaytaradi.
`duration` berilganda `postprocess_output=true` oxirgi jimlikni kesishi mumkin.
`postprocess_output=false` bo'lsa ham standart `pad_duration=0.1` har ikki chetga
jimlik qo'shadi. 6 soniyalik natija uchun quyidagi namunada postprocessing va padding
o'chirilgan. Uzun matnga juda qisqa vaqt berish nutq sifatini pasaytirishi mumkin.

```json
{"text":"Assalomu alaykum. Bu olti soniyalik audio.","language":"uz",
 "voice":"female_1","duration":6.0,
 "generation":{"postprocess_output":false,"pad_duration":0.0,"fade_duration":0.0}}
```

```json
{"text":"I have 123 apples.","language":"en","voice":"female_1",
 "speed":1.2,"seed":123,"normalize_text":true,
 "generation":{"num_step":16,"denoise":true,"postprocess_output":true}}
```

Normalizatsiya: en/zh uchun WeTextProcessing son, sana va valyutani; num2words
qo'llaydigan boshqa tillarda butun sonlarni tayyorlaydi. Aniq ro'yxat:
`normalization_languages`, `normalization_modes`.
**O'zbekcha normalizatsiya qo'llanmaydi:** `language="uz"` bilan `normalize_text=true`
422 qaytaradi. O'zbekcha sonlarni so'z bilan yozing yoki sozlamani `false` qoldiring.
`language=null` bilan ham normalizatsiya yoqilmaydi.

## Speaker tavsifi va matn ichidagi belgilar

`instruct` erkin buyruq emas; vergul bilan ajratilgan qo'llanadigan speaker atributlari:
jins, yosh, pitch, shivirlash, inglizcha aksent yoki xitoycha dialekt.
Misollar: `female, young adult, moderate pitch`; `male, middle-aged, low pitch, whisper`.
To'liq qiymatlar va xitoycha muqobillar: `/v1/capabilities` -> `voice_design_attributes`.
Inglizcha aksent inglizcha nutqqa, xitoycha dialekt xitoycha nutqqa taalluqli.
Voice design asosan en/zh ma'lumotlarida o'qitilgan; boshqa tillardagi sifat alohida tekshiriladi.
Istalgan emotsiya yoki barcha o'zbek shevalarini aniq boshqarish va'da qilinmaydi.

Matn ichida `[laughter]`, `[sigh]` kabi belgilar ishlaydi; ro'yxat capabilities'dagi `nonverbal_tags` da.
Inglizcha talaffuz: `He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.`
Xitoychada belgi o'rniga tone raqamli pinyin, masalan `ZHE2`, beriladi.
Bu fonetik boshqaruv o'zbek fonemalari uchun hujjatlashtirilmagan.

## Audio namunadan klonlash va transkripsiya

Audio davomiyligi va fayl hajmiga standart dasturiy kvota qo'yilmaydi. WAV/FLAC bilan birga telefon va
brauzer yozuvlari: **M4A/AAC, MP3, OGG/Opus, WebM, AMR, 3GP, WMA, AIFF, CAF**
ham qabul qilinadi. Audio trekli MP4/MKV/AVI fayllaridan birinchi audio trek olinadi;
video tasviri modelga yuborilmaydi. Qo'llanadigan kengaytmalar `/v1/capabilities`
ichidagi `reference_audio_formats` da beriladi.

WAV/FLAC namunalarining yaroqli sample rate va kanal soni avtomatik qayta ishlanadi. Qolgan formatlar
serverda avtomatik **24000 Hz mono** audioga aylantiriladi. Telefon yozuvini qo'lda
WAV ga aylantirish yoki kengaytmasini almashtirish shart emas: format faylning asl
mazmunidan aniqlanadi. Bo'sh, jim, buzilgan, ochilmaydigan va DRM bilan himoyalangan
fayllar qabul qilinmaydi. Uzun yozuvdan ovozli namuna avtomatik tanlanadi va aynan shu qism transkripsiya qilinadi. To'liq yozuv matni kesilgan audio bilan aralashtirilmaydi. `reference_preparation` tanlangan qismni ko'rsatadi. Transkripsiya endpointi esa to'liq yozuvni oynalab qayta ishlaydi.
Audio faylining o'zi yuboriladi; tashqi URL yoki server fayl yo'li qabul qilinmaydi.
Frontend `multipart/form-data` ichida klonlash uchun `ref_audio`, transkripsiya
uchun `audio` file maydonidan foydalansin. Agar file picker faqat WAV/FLAC ni ko'rsatsa,
uni `audio/*,.m4a,.aac,.ogg,.opus,.webm,.amr,.3gp,.wma,.aiff,.caf,.mp4,.mkv,.avi`
formatlarini tanlashga moslash kerak.

`POST /v1/audio/clone`, `multipart/form-data`:

| Maydon | Talab |
|---|---|
| `ref_audio` | Majburiy audio fayl. |
| `text` | Majburiy yangi matn, bo'sh bo'lmagan matn. |
| `key` | Ixtiyoriy fayl nomi kaliti: `assistant.welcome` -> `assistant.welcome.wav`; bo'sh yoki berilmasa `speech.wav`. Speech JSON dagi xavfsiz kalit qoidalari amal qiladi. |
| `language` | Ixtiyoriy til kodi. |
| `ref_text` | Ixtiyoriy, namunada aynan gapirilgan matn, bo'sh bo'lmagan matn. Bo'sh qoldirilsa avtomatik transkripsiya ishlaydi. |
| `instruct` | Ixtiyoriy speaker atributlari. Bo'sh qoldirilsa qo'shimcha tavsif berilmaydi. |
| `options` | Ixtiyoriy JSON yozilgan text maydoni: speed/duration/seed/normalize_text/generation; standart `{}`. |

```bash
curl --fail-with-body http://10.2.1.150:8070/v1/audio/clone \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -F "ref_audio=@ref.wav" -F "text=Assalomu alaykum. Bu yangi matn." \
  -F "key=assistant.welcome" \
  -F "language=uz" -F 'options={"speed":1.1,"generation":{"num_step":16}}' \
  --output assistant.welcome.wav
```

Swagger'da `ref_text` va `instruct` boshlang'ich holatda bo'sh turadi. Bu multipart
maydonlardagi bo'sh yoki faqat probeldan iborat qiymatlar yuborilmagan deb olinadi.

`ref_text` berilmasa, server Whisper bilan transkript oladi. `asr_available` mavjudlikni,
`asr_loaded` esa model xotiraga yuklanganini bildiradi. Birinchi ASR so'rovi uzoqroq
bo'lishi mumkin. ASR bo'lmasa, `ref_text` berish kerak; aks holda 503 qaytadi.
Avtotranskript taxminiy: o'zbekcha audio kirill yozuvida yoki xato transkript bilan
chiqishi mumkin. Klonlash sifati uchun namunada aynan aytilgan matnni `ref_text`
orqali aniq berish tavsiya qilinadi.

`POST /v1/audio/transcriptions`: `audio` nomli file maydoni.
Javob: `{"text":"Namunada gapirilgan matn."}`. To'liq audio bo'laklab transkripsiya qilinadi.

## Ovoz profilini saqlash

`POST /v1/voices/clone`, multipart:
`ref_audio`, `name`, `gender` majburiy; `reference_language`, `ref_text` ixtiyoriy;
`preprocess_prompt` bool, standart `true`. Transkript berilmasa ASR ishlatiladi.

`POST /v1/voices/design`, JSON:

```json
{
  "name":"Yangi ayol ovozi",
  "gender":"female",
  "language":"uz",
  "instruct":"female, young adult, moderate pitch",
  "reference_text":"Assalomu alaykum. Bugun siz bilan ravon va tushunarli ovozda suhbatlashamiz.",
  "seed":123,
  "normalize_text":false,
  "generation":{"num_step":32}
}
```

`name`, `gender`, `instruct` majburiy; `language` standarti `uz`; `reference_text`
ixtiyoriy. Profil uchun mos hajmdagi matn avtomatik tanlanib, 8 soniyalik ovoz namunasi yaratiladi.
Bu saqlash endpointi `speed` va `duration` qabul qilmaydi; keyingi speech so'rovida sozlang.

Ikkalasi 201 va profil metama'lumotini qaytaradi: `id`, `name`, `gender`,
`reference_language`, `synthetic`, `source`, `ref_text`, `sample_url`.
`id` ni `voice` sifatida ishlating, namunani `sample_url` orqali tinglang.
Custom ovozlar serverda saqlanadi va qayta ishga tushganda yuklanadi.
Saqlangan klonni qayta ishlatish uchun audio namunani yana yubormang;
`POST /v1/audio/speech` ga qaytgan ID bilan murojaat qiling:

```json
{"key":"assistant.welcome","text":"Davom etish uchun tilni tanlang.","language":"uz","voice":"clone_SERVER_QAYTARGAN_ID"}
```

`voice` ovoz profilini, `key` yuklanadigan fayl nomini tanlaydi. Bir xil `voice`
saqlangan speaker namunasini qayta ishlatadi; har safargi ohang yoki WAV baytlari
aynan bir xil bo'lishi kafolatlanmaydi. `/v1/audio/clone` esa bir martalik generatsiya,
u qayta ishlatiladigan ID saqlamaydi.
Saqlangan ovozlar soni va ularning jami hajmiga dasturiy kvota qo'yilmaydi; diskda joy bo'lishi kerak.

Saqlangan ovozlar bilan ishlash:

1. `POST /v1/voices/clone` orqali audio namunani bir marta yuboring. `name` ga tanish nom bering va qaytgan `id` ni saqlang.
2. `GET /v1/voices` orqali nomi va ID sini toping. Keyingi matnlar uchun `/v1/audio/speech` dagi `voice` maydoniga shu ID ni yuboring; audio qayta yuklanmaydi.
3. Kerak bo'lmasa `DELETE /v1/voices/{voice_id}` ga shu ID ni yuboring. Bu ommaviy profilni, namuna audiosini va profil promptini o'chiradi. Oldin qabul qilingan ishlar uchun xususiy prompt nusxasi ular tugaguncha saqlanadi; tayyor WAV fayllar standart holatda avtomatik o'chirilmaydi.

```bash
curl --fail-with-body -X DELETE http://10.2.1.150:8070/v1/voices/clone_SERVER_QAYTARGAN_ID \
  -H "Authorization: Bearer $TTS_API_KEY"
```

Muvaffaqiyatli javob: `{"id":"clone_SERVER_QAYTARGAN_ID","deleted":true}` (200).
O'chirilgan ovoz ro'yxatdan yo'qoladi, ID qayta ishlamaydi va profil uchun joy bo'shaydi.
Saqlangan clone va design profillari o'chiriladi; `female_1`, `female_2`, `male_1`,
`male_2` uchun 403 qaytadi. Noma'lum yoki allaqachon o'chirilgan ID uchun 404.
O'chirish 503 qaytarsa, shu ID bilan DELETE ni qayta yuboring: profil faol ro'yxatdan
olingan bo'lishi mumkin, lekin fayllarni tozalash hali tugamagan. So'rov timeout
bo'lsa ham operatsiya yakunlanishi mumkin; holatni ro'yxatdan tekshiring.

## Batch ZIP

`POST /v1/audio/batch` -> `application/zip`. Har element speech maydonlarini qabul qiladi.
Elementlar soni va matn uzunligiga standart dasturiy kvota yo'q.

```json
{"items":[
  {"text":"Assalomu alaykum.","language":"uz","voice":"female_1"},
  {"text":"Hello.","language":"en","mode":"design","instruct":"male, young adult"},
  {"text":"This is an automatic voice.","language":"en","mode":"auto"}
]}
```

ZIP ichida `001.wav`, `002.wav` va hokazo, shuningdek `manifest.json` mavjud.
Elementda `key` bo'lsa, uning fayli `key.wav` deb nomlanadi va kalit manifestda saqlanadi.
ZIP ichidagi nomlar harf registridan qat'i nazar takrorlanmasligi kerak; `key="002"`
ikkinchi elementning standart `002.wav` nomi bilan ham to'qnashadi. Nomlar to'qnashsa,
generatsiyadan oldin 422 qaytariladi.
Kirishlardan bittasi noto'g'ri bo'lsa, generatsiya boshlanishidan oldin so'rov rad etiladi.

## Oddiy curl va Python

`request.json` avvalgi oddiy preset so'rovini saqlaydi.

```bash
curl --fail-with-body http://10.2.1.150:8070/v1/languages \
  -H "Authorization: Bearer $TTS_API_KEY"
curl --fail-with-body http://10.2.1.150:8070/v1/audio/speech \
  -H "Authorization: Bearer $TTS_API_KEY" -H "Content-Type: application/json" \
  --data-binary @request.json --output speech.wav
```

PowerShell'da `curl.exe` va kalit uchun `$env:TTS_API_KEY` dan foydalaning.
Python uchun `requests` kerak:

```python
import os
from pathlib import Path
import requests
base_url = "http://10.2.1.150:8070"
headers = {"Authorization": f"Bearer {os.environ['TTS_API_KEY']}"}
voices = requests.get(base_url + "/v1/voices", headers=headers,
                      params={"gender":"female"}, timeout=15)
voices.raise_for_status()
response = requests.post(base_url + "/v1/audio/speech", headers=headers,
    json={"text":"Assalomu alaykum.","language":"uz","gender":"female","voice":"female_1"},
    timeout=200)
response.raise_for_status()
assert response.headers["content-type"].startswith("audio/wav")
Path("speech.wav").write_bytes(response.content)
```

## Resurslar va xatolar

- HTTP body, audio hajmi, matn uzunligi va topshiriqlar soni uchun standart dasturiy kvotalar o'chirilgan. Yuklangan fayllar va PCM `/data/tmp` disk katalogida saqlanib, bo'laklab o'qiladi; 1 GiB RAM tmpfs upload uchun ishlatilmaydi.
- Jobs: barcha matnlar navbatga yoziladi. Qabul qilishda kutiladigan PCM hajmi matn/davomiylik/tezlikdan baholanib, haqiqiy bo'sh disk joyi tekshiriladi; bu generatsiya davomiyligini kesmaydi.
- `TTS_QUEUE_SIZE=0` boshqaruv operatsiyalari kutish navbatiga son kvotasini qo'ymaydi. ASR, profil yaratish/o'chirish va eski clone/batch ishlar speech jarayonlari bilan bir vaqtda modeldan foydalanmaydi.
- Speech uchun `TTS_PARALLEL_WORKERS` o'rinlari ishlaydi; ortiqcha matnlar FIFO navbatda kutadi. Haqiqiy son RAM/GPU zaxirasi sababli sozlangan qiymatdan kam bo'lishi mumkin.
- Jobs POST javobi darhol 202; har bir WAV alohida tayyor bo'ladi. Eski audio/ZIP endpointlari tayyor natijani kutadi; oqimli audio yo'q.
- `TTS_REQUEST_TIMEOUT=0` standart bajarilish muddatini o'chiradi. Natijalar `TTS_JOB_RETENTION_SECONDS=0` bilan avtomatik o'chirilmaydi. Sinxron mijoz/proxy timeouti kutish va generatsiyaga yetishi kerak; ko'p matn uchun jobs endpointini ishlating.
- HTTP timeout yoki uzilish qabul qilingan speech ishini bekor qilmaydi. Profil saqlash operatsiyasi ham davom etishi mumkin; takrorlashdan oldin `/v1/voices` ni tekshiring.
- Uzun yoki tinish belgisiz jumlalar zarur bo'lsa avtomatik mayda segmentlarga ajratiladi. Har bir model segmentining xotira budjeti, fayl tuzilishi, bo'sh/jim audio, parametrlar va xavfsiz fayl nomlari tekshiriladi. Disk/RAM/GPU fizik resurslari cheksiz emas.

| HTTP | Ma'nosi |
|---|---|
| 401 | Kalit noto'g'ri/yuborilmagan. |
| 403 | Tayyor preset ovozini o'chirish taqiqlangan. |
| 404 | Ovoz, ish guruhi yoki saqlangan audio topilmadi. |
| 409 | So'ralgan ish audiosi hali tayyor emas. |
| 413 | Administrator alohida yoqqan body/audio kvotasi oshdi. |
| 422 | Kirish, audio formati, parametr yoki normalizatsiya tili xato. |
| 429 | Boshqaruv operatsiyalari navbati to'la; `Retry-After` soniyadan so'ng qayta yuboring. |
| 503 | Model tayyor emas, kerakli ASR o'rnatilmagan yoki generatsiya xatosi. |
| 504 | Ishning bajarilish yoki boshqaruv so'rovining kutish muddati tugadi. |
| 507 | Speech natijalari uchun disk joyi yetarli emas. |

429/503 holatida kutish va cheklangan qayta urinish ishlating.

## Serverda boshqarish

Loyiha: `C:\text-speak`; Docker Compose loyihasi/konteyner: `text-speak`.
Image: `text-speak:quality-speed-20261107-093515`; xizmat: `api`; port: 8070; serverdagi 0-GPU.
Bitta Uvicorn worker; model/profillarni boshqaruvchi asosiy jarayon va
`spawn` orqali alohida speech jarayonlari. Profillar, SQLite navbat va audiolar:
`text-speak-data` named volume (`/data/voices`, `/data/jobs`).
OmniVoice `offline/model`, Whisper `offline/whisper` dan read-only ulanadi.

```powershell
Set-Location C:\text-speak
docker compose ps
docker compose logs --tail 100 api
docker compose restart api
```

Profil/navbat/audio zaxirasini oling: `docker compose down -v` volume ma'lumotlarini o'chiradi.
Kalitni `.env` dagi `TTS_API_KEY` orqali almashtirib, `docker compose up -d --no-build api` bajaring.
Boshqa brauzer origin'i kerak bo'lsa, `TTS_CORS_ORIGINS` ga aniq originlarni kiriting.
TensorRT image'i tekshirilgan GPU bazasiga offline paketlar o'rnatib yig'ilgan;
ishga tushirishda Compose tayyor image va `text-speak-tensorrt` plan volume'ini ishlatadi.

- OmniVoice model: `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.
- OmniVoice kod: `08be0b4ccbac3e13e374e86fbfead4b4cac343e2`.
- Whisper: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`.
- Model yuklamasdan test: `python -m pytest -q tests/test_api.py tests/test_engine.py`.

### Generatsiyani tezlashtirish

Docker Compose `TTS_ACCELERATION=flashinfer` va `TTS_FLASHINFER_GRAPHS=1` bilan ishlaydi.
`text-speak-tensorrt` tashqi volume tayyor plan/manifestni faqat o'qish uchun ulaydi.
Production Compose tayyor image ishlatadi; oddiy Dockerfile bilan uni qayta yig'maydi.
TensorRT o'rnatilmagan image bilan ishlashda `TTS_ACCELERATION=flashinfer` tanlang.
Oddiy Python/API ishga tushirishda standart backend `none`; kerak bo'lsa shu muhit
o'zgaruvchisini aniq bering. Oddiy backendga qaytish uchun `.env` ga
`TTS_ACCELERATION=none` yozib, `docker compose up -d --no-build api` bajaring.

FlashInfer 32 ta generatsiya qadamini va saqlangan ovoz promptlarini ishlatadi;
tezlashtirish uchun standart qadamlar soni kamaytirilmaydi. Asl shaklga mos CUDA
graflar har ishchida ko'pi bilan 8 ta, 768 MiB taxminiy kesh hisobi bilan saqlanadi;
bu butun jarayon xotirasi yoki capture cho'qqisiga qattiq limit emas. Mos kelmagan
shakl/xotira holati generatsiyadan oldin odatiy FlashInferga o'tadi. Speech har bir alohida jarayonda bajariladi; profil/ASR
operatsiyalari faol speech ishlari tugashini kutib, alohida bajariladi.
Har ishning berilgan `seed` qiymati boshqa speech jarayonidagi ish bilan aralashmaydi;
bu turli qurilma yoki backendlar orasida bitma-bit bir xil WAV va'dasi emas.

Ixtiyoriy TensorRT backend `TTS_ACCELERATION=tensorrt`, `TTS_TENSORRT_PLAN` va
`TTS_TENSORRT_MANIFEST` bilan yoqiladi. Buning uchun TensorRT o'rnatilgan alohida
image va aynan shu checkpoint/GPU/runtime uchun tekshirilgan plan kerak; oddiy
image TensorRT paketini o'z ichiga olmaydi. Manifest plan va model fayllarining
SHA256 qiymatlarini hamda paket versiyalarini tekshiradi. TensorRT bitta segment
uchun CUDA graph saqlaydi, 32 qadamli asl generatsiyani bajaradi. Plan diapazoniga
sig'magan yoki boshqa generatsiya sozlamali vazifalar ish boshlashdan oldin
FlashInferga yuboriladi. Bu matn uzunligi yoki so'rovlar soniga cheklov qo'ymaydi.
TensorRT bajarila boshlagandan keyingi xato yashirilmaydi; xato bergan worker
almashtiriladi va navbatdagi ishlar davom etadi. Profil yaratish va transkripsiya
egasi FlashInferda ishlaydi. `.env` ichida `TTS_ACCELERATION=flashinfer` qilib,
`docker compose up -d --no-build api` bajarish shu image ichidagi FlashInferga qaytaradi.

Compose standart `TTS_PARALLEL_WORKERS=2`, `TTS_JOBS_DIR=/data/jobs` va
`TTS_JOB_RETENTION_SECONDS=0` ni beradi. Asosiy model va speech jarayonlari
birgalikda xotira egallaydi; 16 GiB konteyner RAM limiti va GPU 0 saqlanadi.
Startup RAM/GPU zaxirasini tekshiradi va talab qilingan 2 o'rinni 1 taga kamaytirishi
mumkin. Haqiqiy server sinovisiz 2/3 ta parallel ish sig'ishi yoki ma'lum tezlik
kafolatlanmaydi. Joriy o'rinlar sonini `/v1/audio/jobs` dan tekshiring;
ko'proq HTTP so'rovini qabul qilish uchun Uvicorn'dagi eski 32 ulanish cheklovi olib tashlangan.

Joriy RTX 5090 (SM 12.0), PyTorch 2.10/cu128 muhiti uchun
`FLASHINFER_CUDA_ARCH_LIST=12.0a` ishlatiladi. Bu CUDA 12.8 qo'llaydigan haqiqiy
GPU targetidir; boshqa GPU yoki CUDA muhitiga ko'chirganda targetni qayta tekshiring.
`FLASHINFER_USE_CUDA_NORM=1` CUDA normalizatsiya kernelini aniq tanlaydi;
global o'rnatishdagi CuTe avtomatik tanlovi tasdiqlangan namunadan farq qilishi mumkin.
Bu sozlama importdan oldin beriladi; startup faol kernelni ham tekshiradi.
FlashInfer va uning oldindan yig'ilgan CUDA 12.8 kernel keshi `requirements.lock`
hamda `offline/wheels` ichida versiya va SHA256 bilan biriktirilgan. Yig'ish paytida
PyTorch yoki CUDA runtime almashtirilmaydi.

## Tekshiruv holati

2026-11-07: 578 ta Python testi (1 ta skip), 20 ta frontend testi o'tdi.
Haqiqiy GPU'da 19 ta eager holatga nisbatan 37 ta graf/fallback/qayta-capture
solishtiruvida tokenlar va WAV hashlari teng. Serverda 2 worker bilan 50 ta uzun
matn 77.12 soniyadan 69.39 soniyaga tushdi; barcha 50 ta WAV oldingi hashga teng.
Bu ayni ovoz, matnlar va seed uchun o'lchov; har qanday yuklamaga tezlik kafolati emas.
42 ta ovoz fayli, muhit sozlamalari va boshqa xizmatlar saqlandi.
Dalillar: `verification/quality-speed/assessment.json`, `comparison-50.json`,
`bounded-graph.json` va `private-http.json`.

Quyidagi natijalar avvalgi joylashtirishlarga tegishli. Yangi jobs/navbat va parallel
jarayonlar uchun ular haqiqiy GPU yuk sinovi yoki production joylashtirish dalili emas.

Birinchi joylashtirish, API 1.0 (2026-11-03): 32 ta avtomatik test, to'rtta o'zbekcha
ovoz, inglizcha/ruscha WAV, 1000 belgilik matn va qayta ishga tushgandan keyingi profil
hashlari tekshirilgan. 62.49 soniyalik audio taxminan 5.5 soniyada yaratilgani o'sha sinovga tegishli.

API 2.0 (2026-11-03) serverda `text-speak:2.0.0` sifatida ishga tushirildi.
167 ta avtomatik test o'tdi. Ishlab turgan API'ga boshqa kompyuterdan yuborilgan
24 ta haqiqiy so'rov tekshiruvida preset/design/auto, 6 soniyalik davomiylik,
tezlik, inglizcha normalizatsiya, nonverbal/fonetik boshqaruv, qo'lda va ASR
transkriptli klonlash, transkripsiya, profil yaratish/qayta ishlatish, batch ZIP
va 1000 belgilik matn o'tgan. Klon va design profillari konteyner qayta ishga
tushgach qayta ishlatildi; avvalgi to'rtta ovozning 12 ta fayli o'zgarmadi.
Nomzod konteynerda FLAC/16000 Hz yuklash va xitoycha/ruscha normalizatsiya bilan
ovoz yaratish ham tekshirildi. Natijalar: `verification/full-api/full-api-result.json`.
Talaffuz, tabiiylik va ayol/erkak sifatida eshitilishi tinglab baholanishi kerak;
qo'llanadigan barcha tillar ushbu serverda sinovdan o'tgan deb hisoblanmaydi.

## Litsenziya

OmniVoice kodi Apache 2.0, tayyor modeli CC-BY-NC.
Tijoriy foydalanish uchun mos ruxsat/litsenziya talablarini alohida hal qilish kerak.
[Rasmiy model kartasi](https://huggingface.co/k2-fsa/OmniVoice#license).
Whisper modeli MIT litsenziyasida.
