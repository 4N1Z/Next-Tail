from django.db import models

# Create your models here.
class image(models.Model):
    image = models.ImageField(upload_to='images/')
    def __str__(self):
        return self.image.url
    def geturl(self):
        return self.image.url