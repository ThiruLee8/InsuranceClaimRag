import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { DocumentsComponent } from './documents.component';
import { NotificationService } from '../../core/services/notification.service';

describe('DocumentsComponent', () => {
  let fixture: ComponentFixture<DocumentsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DocumentsComponent, NoopAnimationsModule],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        {
          provide: NotificationService,
          useValue: { success: () => undefined, error: () => undefined, handleHttpError: () => undefined },
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(DocumentsComponent);
  });

  it('creates the document list component', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });
});
