import { HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Observable, throwError } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class NotificationService {
  private readonly snackBar = inject(MatSnackBar);

  success(message: string): void {
    this.snackBar.open(message, 'Close', {
      duration: 3500,
      panelClass: ['snack-success'],
    });
  }

  error(message: string): void {
    this.snackBar.open(message, 'Close', {
      duration: 5000,
      panelClass: ['snack-error'],
    });
  }

  handleHttpError(error: HttpErrorResponse): Observable<never> {
    const apiMessage =
      error.error?.message ||
      (typeof error.error === 'string' ? error.error : null) ||
      error.message ||
      'Unexpected error';
    this.error(apiMessage);
    return throwError(() => error);
  }
}
