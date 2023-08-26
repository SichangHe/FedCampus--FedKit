# Generated by Django 4.2.2 on 2023-08-26 07:51

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("train", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="CoreMLModel",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(editable=False, max_length=64, unique=True)),
                (
                    "file_path",
                    models.CharField(editable=False, max_length=64, unique=True),
                ),
                ("layers_names", models.JSONField(editable=False)),
                (
                    "data_type",
                    models.ForeignKey(
                        editable=False,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="coreml_models",
                        to="train.trainingdatatype",
                    ),
                ),
            ],
        ),
    ]