#!/usr/bin/env python3
"""
Create sample Romanian training data for testing.

This script generates synthetic Romanian instruction-following examples
for quick testing and validation of the training pipeline.

Usage:
    python scripts/create_sample_data.py --num-examples 100
"""

import argparse
import json
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample Romanian training data
SAMPLE_DATA = [
    {
        "instruction": "Care este capitala României?",
        "response": "Capitala României este București, cel mai mare oraș din țară și centrul politic, economic și cultural al națiunii. București este situat în sudul României și este cunoscut pentru arhitectura sa impresionantă și viața culturală vibrantă."
    },
    {
        "instruction": "Explică procesul de fotosinteză.",
        "response": "Fotosinteza este procesul prin care plantele verzi și alte organisme transformă energia luminoasă în energie chimică. În timpul fotosintezei, plantele absorb dioxidul de carbon din aer și apa din sol, folosind lumina soarelui pentru a produce glucoză și oxigen. Acest proces este esențial pentru viața pe Pământ."
    },
    {
        "instruction": "Care sunt cele mai mari orașe din România?",
        "response": "Cele mai mari orașe din România sunt: București (capitala și cel mai mare oraș), Cluj-Napoca (important centru universitar), Timișoara (centru cultural și economic), Iași (oraș istoric și universitar), Constanța (port la Marea Neagră) și Craiova."
    },
    {
        "instruction": "Descrie muntele Carpați.",
        "response": "Munții Carpați formează cel mai mare lanț muntos din România, străbătând țara într-un arc caracteristic. Sunt cunoscuți pentru biodiversitatea lor bogată, peisajele spectaculoase și stațiunile montane. Vârful Moldoveanu este cel mai înalt vârf din România, având 2544 de metri."
    },
    {
        "instruction": "Ce este inteligența artificială?",
        "response": "Inteligența artificială (IA) reprezintă domeniul informaticii care se ocupă cu crearea de sisteme capabile să efectueze sarcini care necesită în mod normal inteligență umană. Acestea includ învățarea automată, recunoașterea vocală, procesarea limbajului natural și luarea deciziilor."
    },
    {
        "instruction": "Explică importanța educației.",
        "response": "Educația este fundamentală pentru dezvoltarea personală și progresul societății. Prin educație, indivizii dobândesc cunoștințe, abilități și valori necesare pentru a avea succes în viață. Educația promovează gândirea critică, creativitatea și capacitatea de a contribui pozitiv la comunitate."
    },
    {
        "instruction": "Ce este Dunărea?",
        "response": "Dunărea este al doilea cel mai lung fluviu din Europa, traversând România pe o distanță de aproximativ 1075 km. Fluviul formează o mare parte din frontiera sudică a țării și se varsă în Marea Neagră prin Delta Dunării, o rezervație naturală unică și protejată UNESCO."
    },
    {
        "instruction": "Descrie importanța Revoluției din 1989.",
        "response": "Revoluția din decembrie 1989 a reprezentat momentul crucial în care România și-a câștigat libertatea de la regimul comunist. Acest eveniment istoric a marcat tranziția către democrație și economia de piață, schimbând fundamental viața românilor și deschizând calea către integrarea europeană."
    },
    {
        "instruction": "Care sunt tradițiile românești de Crăciun?",
        "response": "Tradițiile românești de Crăciun includ colindatul (cântarea de colinde din casă în casă), pregătirea mâncărurilor tradiționale precum sarmale și cozonac, decorarea bradul de Crăciun și participarea la slujbele religioase. Este o perioadă de sărbătoare în familie și comunitate."
    },
    {
        "instruction": "Explică sistemul solar.",
        "response": "Sistemul solar este format din Soare și toate corpurile cerești care orbitează în jurul său, incluzând opt planete majore, planete pitice, luni, asteroizi și comete. Pământul este a treia planetă de la Soare și singura cunoscută până acum care susține viață."
    }
]


def create_sample_dataset(num_examples: int, output_dir: str = "data/raw") -> None:
    """Create sample Romanian training data.

    Args:
        num_examples: Number of examples to generate
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "sample.jsonl"

    logger.info(f"Creating {num_examples} sample Romanian examples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_examples):
            # Cycle through sample data
            sample = SAMPLE_DATA[i % len(SAMPLE_DATA)]

            # Add some variation
            data = {
                'text': f"{sample['instruction']}\n\n{sample['response']}",
                'source': 'sample',
                'metadata': {
                    'title': sample['instruction'][:50],
                    'id': f'sample_{i}'
                }
            }

            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    logger.info(f"Created {num_examples} examples in {output_file}")
    logger.info("\nSample data created successfully!")
    logger.info("Next step: python scripts/prepare_data.py")


def main():
    parser = argparse.ArgumentParser(description="Create sample Romanian training data")
    parser.add_argument('--num-examples', type=int, default=100, help='Number of examples to create')
    parser.add_argument('--output', type=str, default='data/raw', help='Output directory')

    args = parser.parse_args()

    create_sample_dataset(args.num_examples, args.output)


if __name__ == '__main__':
    main()
