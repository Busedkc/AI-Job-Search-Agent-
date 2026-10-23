"""Graph'a her çalıştırmada dışarıdan verilen, değişmeyen veri.

Ana graph ve subgraph ikisi de kullandığı için ayrı dosyada (döngüsel import olmasın).
Varsayılan değerler var: Studio'dan çalıştırırken context girmeden de çalışsın.
"""
from dataclasses import dataclass, field

from shared.config import USER_ID
from shared.resume import load_resume


@dataclass
class Context:
    resume: str = field(default_factory=load_resume)
    user_id: str = USER_ID  # store'da kimin verisi olduğunu belirler