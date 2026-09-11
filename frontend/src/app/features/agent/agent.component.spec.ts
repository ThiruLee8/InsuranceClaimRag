import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { AgentComponent } from './agent.component';
import { NotificationService } from '../../core/services/notification.service';

describe('AgentComponent', () => {
  let fixture: ComponentFixture<AgentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AgentComponent, NoopAnimationsModule],
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

    fixture = TestBed.createComponent(AgentComponent);
    fixture.detectChanges();
  });

  it('creates the component', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('defaults to race mode so the comparison is one click away', () => {
    expect(fixture.componentInstance.mode).toBe('race');
  });

  it('switches to agent mode for injection and gap samples', () => {
    const cmp = fixture.componentInstance;
    cmp.useSample({
      id: 'injection',
      label: 'Poisoned document',
      task: 'Read the notes',
      why: 'hijack',
    });
    expect(cmp.mode).toBe('agent');
    expect(cmp.form.value.task).toBe('Read the notes');
  });

  it('labels injection flags for the timeline', () => {
    const cmp = fixture.componentInstance;
    expect(cmp.flagLabel('indirect_injection')).toContain('injection');
    expect(cmp.failureModeLabel('quiet_give_up')).toContain('give-up');
  });
});
