# router_format.py

def detect_file_type(query: str) -> str | None:
    """
    Phân tích truy vấn để xác định định dạng file cần tìm.
    Trả về: pdf, docx, jpg, mp4,... hoặc None nếu không có đuôi rõ ràng.
    """
    query = query.lower()
    keywords = {
    # 📄 Tài liệu văn bản
    "pdf": ["pdf", "tài liệu pdf", "tập tin pdf",],
    "doc": ["word", "doc", "docx", "tài liệu word", "văn bản word", "file doc", "tệp docx"],
    "odt": ["odt", "open document", "văn bản odt"],
    "rtf": ["rtf", "tài liệu rtf"],
    "txt": ["text", "txt", "text file", "file văn bản", "ghi chú", "note", "tệp txt"],
    "md": ["markdown", "md", "file markdown"],

    # 📊 Bảng tính
    "xls": ["xls", "excel 2003", "file bảng tính cũ"],
    "xlsx": ["xlsx", "excel", "sheet", "spreadsheet", "bảng tính", "file excel", "tệp excel"],
    "csv": ["csv", "comma separated", "bảng dữ liệu", "dữ liệu thô", "file csv", "tệp csv"],
    "ods": ["ods", "open spreadsheet", "bảng tính ods"],

    # 🖼️ Hình ảnh
    "jpg": ["jpg", "jpeg", "ảnh", "image", "hình ảnh", "file jpg", "tệp ảnh"],
    "png": ["png", "ảnh png", "hình ảnh png"],
    "gif": ["gif", "ảnh động", "ảnh gif"],
    "bmp": ["bmp", "bitmap"],
    "svg": ["svg", "ảnh vector", "vector image"],
    "tiff": ["tiff"],
    "webp": ["webp"],

    # 🎞️ Video
    "mp4": ["mp4", "video", "clip", "phim", "file mp4", "tệp video", "xem video"],
    "avi": ["avi", "video avi"],
    "mov": ["mov"],
    "wmv": ["wmv"],
    "flv": ["flv"],
    "mkv": ["mkv"],
    "webm": ["webm"],

    # 🔊 Âm thanh
    "mp3": ["mp3", "audio", "nhạc", "bài hát", "âm thanh", "nghe nhạc", "file mp3"],
    "wav": ["wav", "âm thanh wav"],
    "aac": ["aac"],
    "ogg": ["ogg"],
    "flac": ["flac"],
    "m4a": ["m4a"],

    # 📽️ Slide thuyết trình
    "ppt": ["ppt", "powerpoint", "slide", "bài giảng", "thuyết trình", "bản trình chiếu"],
    "pptx": ["pptx", "powerpoint mới"],
    "odp": ["odp", "open presentation", "trình chiếu odp"],

    # 🗜️ File nén
    "zip": ["zip", "compressed file", "file nén", "nén zip", "tệp zip"],
    "rar": ["rar", "file rar", "giải nén rar"],
    "7z": ["7z", "7zip"],
    "tar": ["tar", "file tar"],
    "gz": ["gz", "gzip", "file gz"],

    # 💻 Mã nguồn
    "py": ["python", ".py", "source code python", "mã python", "code python"],
    "java": ["java", ".java", "source code java"],
    "cpp": ["cpp", "c++", ".cpp", "mã c++", "code cpp"],
    "c": ["c", ".c", "source code c"],
    "js": ["javascript", ".js", "mã js"],
    "html": ["html", "web page", "trang web"],
    "css": ["css", "file css"],

    # 🗃️ Cấu trúc dữ liệu & metadata
    "json": ["json", "data file", "cấu trúc dữ liệu", "file json"],
    "xml": ["xml", "file xml", "dữ liệu xml"],
    "yaml": ["yaml", "yml", "file yaml"],
    "log": ["log", "nhật ký hệ thống", "file log"]
}


    for file_type, kw_list in keywords.items():
        for keyword in kw_list:
            if keyword in query:
                return file_type
    return None
