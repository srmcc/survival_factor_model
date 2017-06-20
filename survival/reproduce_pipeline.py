import click
import pipeline

@click.command()
@click.option('--root_directory', default = '/home/smccurdy/scratch/survival/survival_factor_model/', prompt ='The root directory of the project', help='This is your directory /xx/xx/survival_factor_model/')
@click.option('--disease_type', default= "GBM", prompt='The disease type',
              help="one of [LGG, GBM, LUAD, LUSC, SIM1, SIM2, SIM3, SIM4]")

@click.option('--gold_standard', default= True, prompt='Do gold standard analysis?',
              help="True if [LGG, GBM, LUAD, LUSC] and if external files are downloaded, otherwise False")

def call_pipeline(root_directory, disease_type, gold_standard):
	if gold_standard == 'True':
		gold_standard = True
	elif gold_standard =='False':
		gold_standard = False
	pipeline.pipeline(root_directory, disease_type, gold_standard)

if __name__ == '__main__':
    call_pipeline()