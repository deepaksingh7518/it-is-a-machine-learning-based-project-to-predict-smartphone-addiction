from django.db import models

# Create your models here.

from unittest.util import _MAX_LENGTH
import os

# Create your models here.

class Register(models.Model):
    name=models.CharField(max_length=50,null=True)
    email=models.EmailField(max_length=50,null=True)
    password=models.CharField(max_length=50,null=True)
    age=models.CharField(max_length=50,null=True)
    contact=models.CharField(max_length=50,null=True)
    
class UserPrediction(models.Model):
    user = models.ForeignKey(Register, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # User inputs - now using CharField to store string values
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Others', 'Others')])
    
    # Prediction inputs - storing actual string values from form
    use_phone_for_notes = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    buy_books_from_phone = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    battery_lasts_day = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    run_for_charger = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    worry_about_losing_phone = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    take_phone_to_bathroom = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    use_phone_in_gatherings = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    check_phone_without_notification = models.CharField(max_length=10, choices=[
        ('Never', 'Never'), 
        ('Rarely', 'Rarely'), 
        ('Sometimes', 'Sometimes'), 
        ('Often', 'Often')
    ])
    check_phone_sleep_wake = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    phone_next_while_sleeping = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    check_phone_during_class = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    rely_on_phone_awkward = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    phone_while_tv_eating = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    panic_attack_without_phone = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    use_phone_on_date = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    phone_for_games = models.CharField(max_length=10, choices=[
        ('0 hours', '0 hours'),
        ('1 hours', '1 hour'), 
        ('2 hours', '2 hours'),
        ('3 hours', '3 hours'),
        ('4 hours', '4 hours'),
        ('5 hours', '5+ hours')
    ])
    live_without_phone = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    
    # Prediction results
    prediction_score = models.FloatField()
    addiction_level = models.CharField(max_length=50)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.name} - {self.addiction_level} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
    


class UserHistory(models.Model):
    user = models.ForeignKey(Register, on_delete=models.CASCADE)
    visit_date = models.DateField(auto_now_add=True)
    visit_time = models.TimeField(auto_now_add=True)
    page_visited = models.CharField(max_length=100)
    
    class Meta:
        ordering = ['-visit_date', '-visit_time']




