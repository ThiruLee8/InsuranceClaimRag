import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { DocumentUploadComponent } from './document-upload.component';
import { NotificationService } from '../../../core/services/notification.service';

describe('DocumentUploadComponent', () => {
  let fixture: ComponentFixture<DocumentUploadComponent>;
  let component: DocumentUploadComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DocumentUploadComponent, NoopAnimationsModule],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        {
          provide: NotificationService,
          useValue: { success: () => undefined, error: () => undefined, handleHttpError: () => undefined },
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(DocumentUploadComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('rejects unsupported extensions', () => {
    const file = new File(['x'], 'virus.exe', { type: 'application/octet-stream' });
    component.setFile(file);
    expect(component.selectedFile).toBeNull();
    expect(component.error).toContain('Unsupported');
  });

  it('accepts pdf files', () => {
    const file = new File(['%PDF'], 'claim.pdf', { type: 'application/pdf' });
    component.setFile(file);
    expect(component.selectedFile).toBe(file);
  });
});
